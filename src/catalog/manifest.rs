/*
 * Parseable Server (C) 2022 - 2025 Parseable, Inc.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    path::Path,
};

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeStringArray, StringArray, StringViewArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::DataType;
use itertools::Itertools;
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::{
        metadata::{RowGroupMetaData, SortingColumn},
        reader::FileReader,
    },
};

use crate::metastore::metastore_traits::MetastoreObject;

use super::column::{Column, ExactHashes, ExactValues, TextNgrams, exact_value_hash};

const EXACT_INDEX_FIELDS_ENV: &str = "P_EXACT_INDEX_FIELDS";
const EXACT_INDEX_MAX_VALUES_ENV: &str = "P_EXACT_INDEX_MAX_VALUES";
const DEFAULT_EXACT_INDEX_MAX_VALUES: usize = 4096;
const EXACT_INDEX_MAX_HASHES_ENV: &str = "P_EXACT_INDEX_MAX_HASHES";
const DEFAULT_EXACT_INDEX_MAX_HASHES: usize = 1_000_000;
const TEXT_INDEX_FIELDS_ENV: &str = "P_TEXT_INDEX_FIELDS";
const TEXT_INDEX_MAX_TERMS_ENV: &str = "P_TEXT_INDEX_MAX_TERMS";
const DEFAULT_TEXT_INDEX_MAX_TERMS: usize = 65_536;

struct ExactIndexConfig {
    fields: Vec<String>,
    max_values: usize,
    max_hashes: usize,
}

struct ExactIndexData {
    values: Option<ExactValues>,
    hashes: Option<ExactHashes>,
}

struct TextIndexConfig {
    fields: Vec<String>,
    max_terms: usize,
}

#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde_repr::Serialize_repr,
    serde_repr::Deserialize_repr,
)]
#[repr(u8)]
pub enum SortOrder {
    AscNullsFirst = 0,
    AscNullsLast,
    DescNullsLast,
    #[default]
    DescNullsFirst,
}

pub type SortInfo = (String, SortOrder);
pub const CURRENT_MANIFEST_VERSION: &str = "v1";

/// An entry in a manifest which points to a single file.
/// Additionally, it is meant to store the statistics for the file it
/// points to. Used for pruning file at planning level.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct File {
    pub file_path: String,
    pub num_rows: u64,
    pub file_size: u64,
    pub ingestion_size: u64,
    pub columns: Vec<Column>,
    pub sort_order_id: Vec<SortInfo>,
}

/// A manifest file composed of multiple file entries.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub version: String,
    pub files: Vec<File>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self {
            version: CURRENT_MANIFEST_VERSION.to_string(),
            files: Vec::default(),
        }
    }
}

impl Manifest {
    pub fn apply_change(&mut self, change: File) {
        if let Some(pos) = self
            .files
            .iter()
            .position(|file| file.file_path == change.file_path)
        {
            self.files[pos] = change
        } else {
            self.files.push(change)
        }
    }
}

impl MetastoreObject for Manifest {
    fn get_object_path(&self) -> String {
        unimplemented!()
    }

    fn get_object_id(&self) -> String {
        unimplemented!()
    }
}

pub fn create_from_parquet_file(
    object_store_path: String,
    fs_file_path: &std::path::Path,
) -> anyhow::Result<File> {
    let mut manifest_file = File {
        file_path: object_store_path,
        ..File::default()
    };

    let file = std::fs::File::open(fs_file_path)?;
    manifest_file.file_size = file.metadata()?.len();

    let file = parquet::file::serialized_reader::SerializedFileReader::new(file)?;
    let file_meta = file.metadata().file_metadata();
    let row_groups = file.metadata().row_groups();

    manifest_file.num_rows = file_meta.num_rows() as u64;
    manifest_file.ingestion_size = row_groups
        .iter()
        .fold(0, |acc, x| acc + x.total_byte_size() as u64);

    let mut columns = column_statistics(row_groups);
    if let Some(config) = exact_index_config_from_env() {
        match exact_index_statistics(
            fs_file_path,
            &config.fields,
            config.max_values,
            config.max_hashes,
        ) {
            Ok(exact_indexes) => {
                for (name, exact_index) in exact_indexes {
                    if let Some(column) = columns.get_mut(&name) {
                        column.exact_values = exact_index.values;
                        column.exact_hashes = exact_index.hashes;
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    "failed to collect exact manifest indexes for {:?}: {err}",
                    fs_file_path
                );
            }
        }
    }
    if let Some(config) = text_index_config_from_env() {
        match text_ngram_statistics(fs_file_path, &config.fields, config.max_terms) {
            Ok(text_indexes) => {
                for (name, text_ngrams) in text_indexes {
                    if let Some(column) = columns.get_mut(&name) {
                        column.text_ngrams = Some(text_ngrams);
                    }
                }
            }
            Err(err) => {
                tracing::warn!(
                    "failed to collect text-ngram manifest indexes for {:?}: {err}",
                    fs_file_path
                );
            }
        }
    }
    manifest_file.columns = columns.into_values().collect();
    let mut sort_orders = sort_order(row_groups);
    if let Some(last_sort_order) = sort_orders.pop()
        && sort_orders
            .into_iter()
            .all(|sort_order| sort_order == last_sort_order)
    {
        manifest_file.sort_order_id = last_sort_order;
    }

    Ok(manifest_file)
}

fn sort_order(
    row_groups: &[parquet::file::metadata::RowGroupMetaData],
) -> Vec<Vec<(String, SortOrder)>> {
    let mut sort_orders = Vec::new();
    for row_group in row_groups {
        let sort_order = row_group.sorting_columns().unwrap();
        let sort_order = sort_order
            .iter()
            .map(|sort_order| {
                let SortingColumn {
                    column_idx,
                    descending,
                    nulls_first,
                } = sort_order;
                let col = row_group
                    .column(*column_idx as usize)
                    .column_descr()
                    .path()
                    .string();
                let sort_info = match (descending, nulls_first) {
                    (true, true) => SortOrder::DescNullsFirst,
                    (true, false) => SortOrder::DescNullsLast,
                    (false, true) => SortOrder::AscNullsFirst,
                    (false, false) => SortOrder::AscNullsLast,
                };

                (col, sort_info)
            })
            .collect_vec();

        sort_orders.push(sort_order);
    }
    sort_orders
}

fn column_statistics(row_groups: &[RowGroupMetaData]) -> HashMap<String, Column> {
    let mut columns: HashMap<String, Column> = HashMap::new();
    for row_group in row_groups {
        for col in row_group.columns() {
            let col_name = col.column_descr().path().string();
            if let Some(entry) = columns.get_mut(&col_name) {
                entry.compressed_size += col.compressed_size() as u64;
                entry.uncompressed_size += col.uncompressed_size() as u64;
                if let Some(other) = col.statistics().and_then(|stats| stats.try_into().ok()) {
                    entry.stats = entry.stats.clone().and_then(|this| this.update(other));
                }
            } else {
                columns.insert(
                    col_name.clone(),
                    Column {
                        name: col_name,
                        stats: col.statistics().and_then(|stats| stats.try_into().ok()),
                        exact_values: None,
                        exact_hashes: None,
                        text_ngrams: None,
                        uncompressed_size: col.uncompressed_size() as u64,
                        compressed_size: col.compressed_size() as u64,
                    },
                );
            }
        }
    }
    columns
}

fn exact_index_config_from_env() -> Option<ExactIndexConfig> {
    let fields = std::env::var(EXACT_INDEX_FIELDS_ENV).ok()?;
    let fields: Vec<_> = fields
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();

    if fields.is_empty() {
        return None;
    }

    let max_values = std::env::var(EXACT_INDEX_MAX_VALUES_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_EXACT_INDEX_MAX_VALUES);

    let max_hashes = std::env::var(EXACT_INDEX_MAX_HASHES_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_EXACT_INDEX_MAX_HASHES);

    (max_values > 0 || max_hashes > 0).then_some(ExactIndexConfig {
        fields,
        max_values,
        max_hashes,
    })
}

fn text_index_config_from_env() -> Option<TextIndexConfig> {
    let fields = std::env::var(TEXT_INDEX_FIELDS_ENV).ok()?;
    let fields: Vec<_> = fields
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect();

    if fields.is_empty() {
        return None;
    }

    let max_terms = std::env::var(TEXT_INDEX_MAX_TERMS_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_TEXT_INDEX_MAX_TERMS);

    (max_terms > 0).then_some(TextIndexConfig { fields, max_terms })
}

fn exact_index_statistics(
    fs_file_path: &Path,
    fields: &[String],
    max_values: usize,
    max_hashes: usize,
) -> anyhow::Result<HashMap<String, ExactIndexData>> {
    let file = std::fs::File::open(fs_file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let projected_indices: Vec<_> = fields
        .iter()
        .filter_map(|field| schema.index_of(field).ok())
        .collect();

    if projected_indices.is_empty() {
        return Ok(HashMap::new());
    }

    let projection = ProjectionMask::roots(builder.parquet_schema(), projected_indices);
    let mut reader = builder.with_projection(projection).build()?;
    let mut values_by_field: HashMap<String, BTreeSet<String>> = fields
        .iter()
        .map(|field| (field.clone(), BTreeSet::new()))
        .collect();
    let mut hashes_by_field: HashMap<String, BTreeSet<u64>> = fields
        .iter()
        .map(|field| (field.clone(), BTreeSet::new()))
        .collect();
    let mut values_complete_by_field: HashMap<String, bool> = fields
        .iter()
        .map(|field| (field.clone(), max_values > 0))
        .collect();
    let mut hashes_complete_by_field: HashMap<String, bool> = fields
        .iter()
        .map(|field| (field.clone(), max_hashes > 0))
        .collect();
    let mut seen_fields = HashSet::new();

    for batch in &mut reader {
        let batch = batch?;
        let schema = batch.schema();
        for (idx, field) in schema.fields().iter().enumerate() {
            let name = field.name();
            let values_active = values_complete_by_field
                .get(name.as_str())
                .copied()
                .unwrap_or(false);
            let hashes_active = hashes_complete_by_field
                .get(name.as_str())
                .copied()
                .unwrap_or(false);
            if !values_active && !hashes_active {
                continue;
            }

            seen_fields.insert(name.clone());
            let Some(values) = values_by_field.get_mut(name.as_str()) else {
                continue;
            };
            let Some(hashes) = hashes_by_field.get_mut(name.as_str()) else {
                continue;
            };

            let (values_complete, hashes_complete) = collect_exact_indexes(
                batch.column(idx).as_ref(),
                values,
                values_active,
                max_values,
                hashes,
                hashes_active,
                max_hashes,
            );
            if !values_complete {
                values.clear();
                values_complete_by_field.insert(name.clone(), false);
            }
            if !hashes_complete {
                hashes.clear();
                hashes_complete_by_field.insert(name.clone(), false);
            }
        }
    }

    Ok(values_by_field
        .into_iter()
        .filter_map(|(name, values)| {
            if !seen_fields.contains(&name) {
                return None;
            }

            let values = values_complete_by_field
                .get(&name)
                .copied()
                .unwrap_or(false)
                .then(|| ExactValues {
                    complete: true,
                    values: values.into_iter().collect(),
                });
            let hashes = hashes_complete_by_field
                .get(&name)
                .copied()
                .unwrap_or(false)
                .then(|| ExactHashes {
                    complete: true,
                    hashes: hashes_by_field
                        .remove(&name)
                        .unwrap_or_default()
                        .into_iter()
                        .collect(),
                });

            (values.is_some() || hashes.is_some())
                .then_some((name, ExactIndexData { values, hashes }))
        })
        .collect())
}

fn collect_exact_indexes(
    array: &dyn Array,
    values: &mut BTreeSet<String>,
    values_active: bool,
    max_values: usize,
    hashes: &mut BTreeSet<u64>,
    hashes_active: bool,
    max_hashes: usize,
) -> (bool, bool) {
    let mut values_complete = values_active;
    let mut hashes_complete = hashes_active;

    for idx in 0..array.len() {
        if array.is_null(idx) {
            continue;
        }

        let Some(value) = exact_value_from_array(array, idx) else {
            return (false, false);
        };
        if values_complete {
            values.insert(value.clone());

            if values.len() > max_values {
                values.clear();
                values_complete = false;
            }
        }
        if hashes_complete {
            hashes.insert(exact_value_hash(&value));

            if hashes.len() > max_hashes {
                hashes.clear();
                hashes_complete = false;
            }
        }

        if !values_complete && !hashes_complete {
            break;
        }
    }

    (values_complete, hashes_complete)
}

fn exact_value_from_array(array: &dyn Array, idx: usize) -> Option<String> {
    macro_rules! downcast_value {
        ($array_ty:ty) => {
            array
                .as_any()
                .downcast_ref::<$array_ty>()
                .map(|array| array.value(idx).to_string())
        };
    }

    match array.data_type() {
        DataType::Utf8 => downcast_value!(StringArray),
        DataType::LargeUtf8 => downcast_value!(LargeStringArray),
        DataType::Utf8View => downcast_value!(StringViewArray),
        DataType::Boolean => downcast_value!(BooleanArray),
        DataType::Int8 => downcast_value!(Int8Array),
        DataType::Int16 => downcast_value!(Int16Array),
        DataType::Int32 => downcast_value!(Int32Array),
        DataType::Int64 => downcast_value!(Int64Array),
        DataType::UInt8 => downcast_value!(UInt8Array),
        DataType::UInt16 => downcast_value!(UInt16Array),
        DataType::UInt32 => downcast_value!(UInt32Array),
        DataType::UInt64 => downcast_value!(UInt64Array),
        DataType::Float32 => downcast_value!(Float32Array),
        DataType::Float64 => downcast_value!(Float64Array),
        _ => None,
    }
}

fn text_ngram_statistics(
    fs_file_path: &Path,
    fields: &[String],
    max_terms: usize,
) -> anyhow::Result<HashMap<String, TextNgrams>> {
    let file = std::fs::File::open(fs_file_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema();
    let projected_indices: Vec<_> = fields
        .iter()
        .filter_map(|field| schema.index_of(field).ok())
        .collect();

    if projected_indices.is_empty() {
        return Ok(HashMap::new());
    }

    let projection = ProjectionMask::roots(builder.parquet_schema(), projected_indices);
    let mut reader = builder.with_projection(projection).build()?;
    let mut grams_by_field: HashMap<String, BTreeSet<String>> = fields
        .iter()
        .map(|field| (field.clone(), BTreeSet::new()))
        .collect();
    let mut complete_by_field: HashMap<String, bool> =
        fields.iter().map(|field| (field.clone(), true)).collect();
    let mut seen_fields = HashSet::new();

    for batch in &mut reader {
        let batch = batch?;
        let schema = batch.schema();
        for (idx, field) in schema.fields().iter().enumerate() {
            let name = field.name();
            if !complete_by_field
                .get(name.as_str())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }

            seen_fields.insert(name.clone());
            let Some(grams) = grams_by_field.get_mut(name.as_str()) else {
                continue;
            };

            if !collect_text_ngrams(batch.column(idx).as_ref(), grams, max_terms) {
                grams.clear();
                complete_by_field.insert(name.clone(), false);
            }
        }
    }

    Ok(grams_by_field
        .into_iter()
        .filter_map(|(name, grams)| {
            if seen_fields.contains(&name) && complete_by_field.get(&name).copied().unwrap_or(false)
            {
                Some((
                    name,
                    TextNgrams {
                        complete: true,
                        min_len: 1,
                        grams: grams.into_iter().collect(),
                    },
                ))
            } else {
                None
            }
        })
        .collect())
}

fn collect_text_ngrams(array: &dyn Array, grams: &mut BTreeSet<String>, max_terms: usize) -> bool {
    for idx in 0..array.len() {
        if array.is_null(idx) {
            continue;
        }

        let Some(value) = string_value_from_array(array, idx) else {
            return false;
        };
        insert_lowercase_pruning_ngrams(value.as_ref(), grams);

        if grams.len() > max_terms {
            return false;
        }
    }

    true
}

fn string_value_from_array(array: &dyn Array, idx: usize) -> Option<String> {
    match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|array| array.value(idx).to_string()),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|array| array.value(idx).to_string()),
        DataType::Utf8View => array
            .as_any()
            .downcast_ref::<StringViewArray>()
            .map(|array| array.value(idx).to_string()),
        _ => None,
    }
}

fn insert_lowercase_pruning_ngrams(value: &str, grams: &mut BTreeSet<String>) {
    let chars = value.to_lowercase().chars().collect::<Vec<_>>();
    for width in 1..=chars.len().min(3) {
        for window in chars.windows(width) {
            grams.insert(window.iter().collect());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use super::{exact_index_statistics, text_ngram_statistics};

    #[test]
    fn exact_value_statistics_reads_dotted_string_fields() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let parquet_path = temp_dir.path().join("events.parquet");
        let schema = Arc::new(Schema::new(vec![
            Field::new("attributes.trace.id.synthetic", DataType::Utf8, true),
            Field::new("body", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["trace-b", "trace-a", "trace-b"])),
                Arc::new(StringArray::from(vec!["one", "two", "three"])),
            ],
        )?;

        let file = std::fs::File::create(&parquet_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        let indexes = exact_index_statistics(
            &parquet_path,
            &["attributes.trace.id.synthetic".to_string()],
            8,
            8,
        )?;
        let exact_values = indexes
            .get("attributes.trace.id.synthetic")
            .and_then(|index| index.values.as_ref())
            .expect("dotted field value index");
        let exact_hashes = indexes
            .get("attributes.trace.id.synthetic")
            .and_then(|index| index.hashes.as_ref())
            .expect("dotted field hash index");

        assert!(exact_values.complete);
        assert_eq!(exact_values.values, vec!["trace-a", "trace-b"]);
        assert!(exact_hashes.complete);
        assert_eq!(exact_hashes.hashes.len(), 2);
        assert!(exact_hashes.contains_value("trace-a"));

        let truncated = exact_index_statistics(
            &parquet_path,
            &["attributes.trace.id.synthetic".to_string()],
            1,
            8,
        )?;
        let truncated = truncated
            .get("attributes.trace.id.synthetic")
            .expect("hash fallback remains after value truncation");
        assert!(truncated.values.is_none());
        assert!(truncated.hashes.is_some());

        Ok(())
    }

    #[test]
    fn text_ngram_statistics_reads_body_substrings() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir()?;
        let parquet_path = temp_dir.path().join("events.parquet");
        let schema = Arc::new(Schema::new(vec![Field::new("body", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![
                "upstream_timeout while loading cart_id",
                "ValueError from large_payload",
            ]))],
        )?;

        let file = std::fs::File::create(&parquet_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        let indexes = text_ngram_statistics(&parquet_path, &["body".to_string()], 1024)?;
        let ngrams = indexes.get("body").expect("body ngram index");

        assert!(ngrams.complete);
        assert!(ngrams.grams.contains(&"ups".to_string()));
        assert!(ngrams.grams.contains(&"u".to_string()));
        assert!(ngrams.grams.contains(&"up".to_string()));
        assert!(ngrams.grams.contains(&"lo".to_string()));
        assert!(ngrams.grams.contains(&"tim".to_string()));
        assert!(ngrams.grams.contains(&"val".to_string()));
        assert_eq!(ngrams.min_len, 1);

        let truncated = text_ngram_statistics(&parquet_path, &["body".to_string()], 1)?;
        assert!(!truncated.contains_key("body"));

        Ok(())
    }
}
