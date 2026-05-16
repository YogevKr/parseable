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

use std::{any::Any, collections::HashMap, ops::Bound, sync::Arc};

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef, SortOptions};
use chrono::{DateTime, NaiveDateTime, TimeDelta, Timelike, Utc};
use datafusion::{
    catalog::{SchemaProvider, Session},
    common::{
        Constraints, ToDFSchema,
        stats::Precision,
        tree_node::{TreeNode, TreeNodeRecursion},
    },
    datasource::{
        MemTable, TableProvider,
        file_format::{FileFormat, parquet::ParquetFormat},
        listing::PartitionedFile,
        physical_plan::{FileGroup, FileScanConfigBuilder, ParquetSource},
    },
    error::{DataFusionError, Result as DataFusionResult},
    execution::object_store::ObjectStoreUrl,
    logical_expr::{
        Between, BinaryExpr, Operator, TableProviderFilterPushDown, TableType, utils::conjunction,
    },
    physical_expr::{LexOrdering, PhysicalSortExpr, create_physical_expr, expressions::col},
    physical_plan::{ExecutionPlan, Statistics, empty::EmptyExec, union::UnionExec},
    prelude::Expr,
    scalar::ScalarValue,
};
use futures_util::TryFutureExt;
use itertools::Itertools;

use crate::{
    catalog::{
        ManifestFile, Snapshot as CatalogSnapshot,
        column::{Column, TypedStatistics},
        manifest::File,
        snapshot::{ManifestItem, Snapshot},
    },
    event::DEFAULT_TIMESTAMP_KEY,
    hottier::HotTierManager,
    metrics::{QUERY_CACHE_HIT, increment_files_scanned_in_query_by_date},
    option::Mode,
    parseable::{DEFAULT_TENANT, PARSEABLE, STREAM_EXISTS},
    storage::{ObjectStorage, ObjectStoreFormat},
};

use super::listing_table_builder::ListingTableBuilder;

// schema provider for stream based on global data
#[derive(Debug)]
pub struct GlobalSchemaProvider {
    pub storage: Arc<dyn ObjectStorage>,
    pub tenant_id: Option<String>,
}

#[async_trait::async_trait]
impl SchemaProvider for GlobalSchemaProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_names(&self) -> Vec<String> {
        PARSEABLE.streams.list(&self.tenant_id)
    }

    async fn table(&self, name: &str) -> DataFusionResult<Option<Arc<dyn TableProvider>>> {
        if self.table_exist(name) {
            Ok(Some(Arc::new(StandardTableProvider {
                schema: PARSEABLE
                    .get_stream(name, &self.tenant_id)
                    .expect(STREAM_EXISTS)
                    .get_schema(),
                stream: name.to_owned(),
                tenant_id: self.tenant_id.clone(),
            })))
        } else {
            Ok(None)
        }
    }

    fn table_exist(&self, name: &str) -> bool {
        PARSEABLE.get_stream(name, &self.tenant_id).is_ok()
    }
}

#[derive(Debug)]
struct StandardTableProvider {
    schema: SchemaRef,
    // prefix under which to find snapshot
    stream: String,
    tenant_id: Option<String>,
}

impl StandardTableProvider {
    #[allow(clippy::too_many_arguments)]
    async fn create_parquet_physical_plan(
        &self,
        execution_plans: &mut Vec<Arc<dyn ExecutionPlan>>,
        object_store_url: ObjectStoreUrl,
        partitions: Vec<Vec<PartitionedFile>>,
        statistics: Statistics,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        state: &dyn Session,
        time_partition: Option<String>,
    ) -> Result<(), DataFusionError> {
        let filters = if let Some(expr) = conjunction(filters.to_vec()) {
            let table_df_schema = self.schema.as_ref().clone().to_dfschema()?;
            let filters = create_physical_expr(&expr, &table_df_schema, state.execution_props())?;
            Some(filters)
        } else {
            None
        };

        let sort_expr = PhysicalSortExpr {
            expr: if let Some(time_partition) = time_partition {
                col(&time_partition, &self.schema)?
            } else {
                col(DEFAULT_TIMESTAMP_KEY, &self.schema)?
            },
            options: SortOptions {
                descending: true,
                nulls_first: true,
            },
        };

        let file_format = ParquetFormat::default().with_enable_pruning(true);

        // create file groups from vec file partitions
        let file_groups = partitions.into_iter().map(FileGroup::new).collect_vec();

        // parquet file source, default table parquet options
        let file_source = if let Some(phyiscal_expr) = filters {
            ParquetSource::new(self.schema.clone()).with_predicate(phyiscal_expr)
        } else {
            ParquetSource::new(self.schema.clone())
        };

        let mut conf_builder = FileScanConfigBuilder::new(object_store_url, file_source.into())
            .with_statistics(statistics)
            .with_batch_size(Some(parquet_scan_batch_size(
                PARSEABLE.options.execution_batch_size,
                limit,
            )))
            .with_constraints(Constraints::default())
            .with_file_groups(file_groups)
            .with_output_ordering(vec![LexOrdering::new([sort_expr]).unwrap()]);

        // Set projection if provided
        if let Some(proj_indices) = projection {
            conf_builder = conf_builder.with_projection_indices(Some(proj_indices.clone()))?;
        }

        // Set limit if provided
        if let Some(lim) = limit {
            conf_builder = conf_builder.with_limit(Some(lim));
        }

        let conf = conf_builder.build();

        // create the execution plan
        let plan = file_format
            .create_physical_plan(
                state,
                // .as_any().downcast_ref::<SessionState>().unwrap(), // Remove this when ParquetFormat catches up
                conf,
            )
            .await?;

        execution_plans.push(plan);

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn get_hottier_exectuion_plan(
        &self,
        execution_plans: &mut Vec<Arc<dyn ExecutionPlan>>,
        hot_tier_manager: &HotTierManager,
        manifest_files: &mut Vec<File>,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        state: &dyn Session,
        time_partition: Option<String>,
    ) -> Result<(), DataFusionError> {
        let hot_tier_files = hot_tier_manager
            .get_hot_tier_manifest_files(manifest_files)
            .await
            .map_err(|err| DataFusionError::External(Box::new(err)))?;

        let hot_tier_files: Vec<File> = hot_tier_files
            .into_iter()
            .map(|mut file| {
                let path = PARSEABLE
                    .options
                    .hot_tier_storage_path
                    .as_ref()
                    .unwrap()
                    .join(&file.file_path);
                file.file_path = path.to_str().unwrap().to_string();
                file
            })
            .collect();

        let (partitioned_files, statistics) = self.partitioned_files(hot_tier_files);

        let object_store_url = "file:///";
        self.create_parquet_physical_plan(
            execution_plans,
            ObjectStoreUrl::parse(object_store_url).unwrap(),
            partitioned_files,
            statistics,
            projection,
            filters,
            limit,
            state,
            time_partition.clone(),
        )
        .await?;

        Ok(())
    }

    /// Create an execution plan over the records in arrows and parquet that are still in staging, awaiting push to object storage
    async fn get_staging_execution_plan(
        &self,
        execution_plans: &mut Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        state: &dyn Session,
        time_partition: Option<&String>,
    ) -> Result<(), DataFusionError> {
        let Ok(staging) = PARSEABLE.get_stream(&self.stream, &self.tenant_id) else {
            return Ok(());
        };

        // Staging arrow exection plan
        let records = staging.recordbatches_cloned(&self.schema);
        let arrow_exec = reversed_mem_table(records, self.schema.clone())?
            .scan(state, projection, filters, limit)
            .await?;
        execution_plans.push(arrow_exec);

        // Get a list of parquet files still in staging, order by filename
        let mut parquet_files = staging.parquet_files();
        parquet_files.sort_by(|a, b| a.cmp(b).reverse());

        // NOTE: We don't partition among CPUs to ensure consistent results.
        // i.e. We were seeing in-consistent ordering when querying over parquets in staging.
        let mut partitioned_files = Vec::with_capacity(parquet_files.len());
        for file_path in parquet_files {
            let Ok(file_meta) = file_path.metadata() else {
                continue;
            };
            let file = PartitionedFile::new(file_path.display().to_string(), file_meta.len());
            partitioned_files.push(file)
        }

        // // NOTE: There is the possibility of a parquet file being pushed to object store
        // // and deleted from staging in the time it takes for datafusion to get to it.
        // // Staging parquet execution plan
        // let object_store_url = if let Some(tenant_id) = self.tenant_id.as_ref() {
        //     &format!("file://{tenant_id}/")
        // } else {
        //     "file:///"
        // };
        let object_store_url = "file:///";
        self.create_parquet_physical_plan(
            execution_plans,
            ObjectStoreUrl::parse(object_store_url).unwrap(),
            vec![partitioned_files],
            Statistics::new_unknown(&self.schema),
            projection,
            filters,
            limit,
            state,
            time_partition.cloned(),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn legacy_listing_table(
        &self,
        execution_plans: &mut Vec<Arc<dyn ExecutionPlan>>,
        glob_storage: Arc<dyn ObjectStorage>,
        time_filters: &[PartialTimeFilter],
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
        time_partition: Option<String>,
    ) -> Result<(), DataFusionError> {
        ListingTableBuilder::new(self.stream.to_owned())
            .populate_via_listing(glob_storage.clone(), time_filters)
            .and_then(|builder| async {
                let table = builder.build(
                    self.schema.clone(),
                    |x| glob_storage.query_prefixes(x),
                    time_partition,
                )?;
                if let Some(table) = table {
                    let plan = table.scan(state, projection, filters, limit).await?;
                    execution_plans.push(plan);
                }

                Ok(())
            })
            .await?;

        Ok(())
    }

    fn final_plan(
        &self,
        mut execution_plans: Vec<Arc<dyn ExecutionPlan>>,
        projection: Option<&Vec<usize>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let exec: Arc<dyn ExecutionPlan> = if execution_plans.is_empty() {
            let schema = match projection {
                Some(projection) => Arc::new(self.schema.project(projection)?),
                None => self.schema.to_owned(),
            };
            Arc::new(EmptyExec::new(schema))
        } else if execution_plans.len() == 1 {
            execution_plans.pop().unwrap()
        } else {
            UnionExec::try_new(execution_plans)?
        };
        Ok(exec)
    }

    fn partitioned_files(
        &self,
        manifest_files: Vec<File>,
    ) -> (Vec<Vec<PartitionedFile>>, datafusion::common::Statistics) {
        let target_partition: usize = num_cpus::get();
        let mut partitioned_files = Vec::from_iter((0..target_partition).map(|_| Vec::new()));
        let mut column_statistics = HashMap::<String, Option<TypedStatistics>>::new();
        let mut count = 0;
        let mut file_count = 0u64;
        for (index, file) in manifest_files
            .into_iter()
            .enumerate()
            .map(|(x, y)| (x % target_partition, y))
        {
            #[allow(unused_mut)]
            let File {
                mut file_path,
                num_rows,
                columns,
                ..
            } = file;

            // Track billing metrics for files scanned in query
            file_count += 1;

            // object_store::path::Path doesn't automatically deal with Windows path separators
            // to do that, we are using from_absolute_path() which takes into consideration the underlying filesystem
            // before sending the file path to PartitionedFile
            // the github issue- https://github.com/parseablehq/parseable/issues/824
            // For some reason, the `from_absolute_path()` doesn't work for macos, hence the ugly solution
            // TODO: figure out an elegant solution to this
            #[cfg(windows)]
            {
                if PARSEABLE.storage.name() == "drive" {
                    file_path = object_store::path::Path::from_absolute_path(file_path)
                        .unwrap()
                        .to_string();
                }
            }
            let pf = PartitionedFile::new(file_path, file.file_size);
            partitioned_files[index].push(pf);

            columns.into_iter().for_each(|col| {
                column_statistics
                    .entry(col.name)
                    .and_modify(|x| {
                        if let Some((stats, col_stats)) = x.as_ref().cloned().zip(col.stats.clone())
                        {
                            // update() returns None on type mismatch (e.g. column
                            // historically written as both Utf8 and Timestamp(ms)).
                            // Dropping to None here makes the planner skip min/max
                            // pushdown for this column instead of crashing the worker.
                            *x = stats.update(col_stats);
                        }
                    })
                    .or_insert_with(|| col.stats.as_ref().cloned());
            });
            count += num_rows;
        }
        let statistics = self
            .schema
            .fields()
            .iter()
            .map(|field| {
                column_statistics
                    .get(field.name())
                    .and_then(|stats| stats.as_ref())
                    .and_then(|stats| stats.clone().min_max_as_scalar(field.data_type()))
                    .map(|(min, max)| datafusion::common::ColumnStatistics {
                        null_count: Precision::Absent,
                        max_value: Precision::Exact(max),
                        min_value: Precision::Exact(min),
                        distinct_count: Precision::Absent,
                        sum_value: Precision::Absent,
                        byte_size: Precision::Absent,
                    })
                    .unwrap_or_default()
            })
            .collect();

        let statistics = datafusion::common::Statistics {
            num_rows: Precision::Exact(count as usize),
            total_byte_size: Precision::Absent,
            column_statistics: statistics,
        };

        // Track billing metrics for query scan
        let current_date = chrono::Utc::now().date_naive().to_string();
        increment_files_scanned_in_query_by_date(
            file_count,
            &current_date,
            self.tenant_id.as_deref().unwrap_or(DEFAULT_TENANT),
        );

        (partitioned_files, statistics)
    }
}

async fn collect_from_snapshot(
    snapshot: &Snapshot,
    time_filters: &[PartialTimeFilter],
    filters: &[Expr],
    limit: Option<usize>,
    stream_name: &str,
    tenant_id: &Option<String>,
) -> Result<Vec<File>, DataFusionError> {
    let mut manifest_files = Vec::new();

    for manifest_item in snapshot.manifests(time_filters) {
        let manifest_opt = PARSEABLE
            .metastore
            .get_manifest(
                stream_name,
                manifest_item.time_lower_bound,
                manifest_item.time_upper_bound,
                Some(manifest_item.manifest_path),
                tenant_id,
            )
            .await
            .map_err(|e| DataFusionError::Plan(e.to_string()))?;
        if let Some(manifest) = manifest_opt {
            manifest_files.push(manifest);
        } else {
            tracing::warn!(
                "Manifest missing for stream={} [{:?} - {:?}]",
                stream_name,
                manifest_item.time_lower_bound,
                manifest_item.time_upper_bound
            );
        }
    }

    let mut manifest_files: Vec<_> = manifest_files
        .into_iter()
        .flat_map(|file| file.files)
        .rev()
        .collect();
    for filter in filters {
        manifest_files.retain(|file| !file.can_be_pruned(filter))
    }
    if let Some(limit) = limit {
        let limit = limit as u64;
        let mut curr_limit = 0;
        let mut pos = None;

        for (index, file) in manifest_files.iter().enumerate() {
            curr_limit += file.num_rows();
            if curr_limit >= limit {
                pos = Some(index);
                break;
            }
        }

        if let Some(pos) = pos {
            manifest_files.truncate(pos + 1);
        }
    }

    Ok(manifest_files)
}

#[async_trait::async_trait]
impl TableProvider for StandardTableProvider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let mut execution_plans = vec![];
        let glob_storage = PARSEABLE.storage.get_object_store();

        let object_store_format: ObjectStoreFormat = serde_json::from_slice(
            &PARSEABLE
                .metastore
                .get_stream_json(&self.stream, false, &self.tenant_id)
                .await
                .map_err(|e| DataFusionError::Plan(e.to_string()))?,
        )
        .map_err(|e| DataFusionError::Plan(e.to_string()))?;

        let time_partition = object_store_format.time_partition;
        let mut time_filters = extract_primary_filter(filters, &time_partition);
        if is_within_staging_window(&time_filters) {
            self.get_staging_execution_plan(
                &mut execution_plans,
                projection,
                filters,
                limit,
                state,
                time_partition.as_ref(),
            )
            .await?;
        };
        let mut merged_snapshot = Snapshot::default();
        if PARSEABLE.options.mode == Mode::Query || PARSEABLE.options.mode == Mode::Prism {
            let obs = PARSEABLE
                .metastore
                .get_all_stream_jsons(&self.stream, None, &self.tenant_id)
                .await;
            if let Ok(obs) = obs {
                for ob in obs {
                    if let Ok(object_store_format) =
                        serde_json::from_slice::<ObjectStoreFormat>(&ob)
                    {
                        let snapshot = object_store_format.snapshot;
                        for manifest in snapshot.manifest_list {
                            merged_snapshot.manifest_list.push(manifest);
                        }
                    }
                }
            }
        } else {
            merged_snapshot = object_store_format.snapshot;
        }

        // Is query timerange is overlapping with older data.
        // if true, then get listing table time filters and execution plan separately
        if is_overlapping_query(&merged_snapshot.manifest_list, &time_filters) {
            let listing_time_fiters =
                return_listing_time_filters(&merged_snapshot.manifest_list, &mut time_filters);

            if let Some(listing_time_filter) = listing_time_fiters {
                self.legacy_listing_table(
                    &mut execution_plans,
                    glob_storage.clone(),
                    &listing_time_filter,
                    state,
                    projection,
                    filters,
                    limit,
                    time_partition.clone(),
                )
                .await?;
            }
        }

        let mut manifest_files = collect_from_snapshot(
            &merged_snapshot,
            &time_filters,
            filters,
            limit,
            &self.stream,
            &self.tenant_id,
        )
        .await?;

        if manifest_files.is_empty() {
            return self.final_plan(execution_plans, projection);
        }

        // Hot tier data fetch
        if let Some(hot_tier_manager) = HotTierManager::global()
            && hot_tier_manager.check_stream_hot_tier_exists(&self.stream, &self.tenant_id)
        {
            self.get_hottier_exectuion_plan(
                &mut execution_plans,
                hot_tier_manager,
                &mut manifest_files,
                projection,
                filters,
                limit,
                state,
                time_partition.clone(),
            )
            .await?;
        }
        if manifest_files.is_empty() {
            QUERY_CACHE_HIT
                .with_label_values(&[
                    &self.stream,
                    self.tenant_id.as_deref().unwrap_or(DEFAULT_TENANT),
                ])
                .inc();
            return self.final_plan(execution_plans, projection);
        }

        let (partitioned_files, statistics) = self.partitioned_files(manifest_files);

        let object_store_url = glob_storage.store_url();

        self.create_parquet_physical_plan(
            &mut execution_plans,
            ObjectStoreUrl::parse(object_store_url).unwrap(),
            partitioned_files,
            statistics,
            projection,
            filters,
            limit,
            state,
            time_partition.clone(),
        )
        .await?;

        Ok(self.final_plan(execution_plans, projection)?)
    }

    /*
    Updated the function signature (and name)
    Now it handles multiple filters
    */
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>, DataFusionError> {
        let res_vec = filters
            .iter()
            .map(|filter| {
                if expr_in_boundary(filter) {
                    // if filter can be handled by time partiton pruning, it is exact
                    TableProviderFilterPushDown::Exact
                } else {
                    // otherwise, we still might be able to handle the filter with file
                    // level mechanisms such as Parquet row group pruning.
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect_vec();
        Ok(res_vec)
    }
}

fn reversed_mem_table(
    mut records: Vec<RecordBatch>,
    schema: Arc<Schema>,
) -> Result<MemTable, DataFusionError> {
    records[..].reverse();
    records
        .iter_mut()
        .for_each(|batch| *batch = crate::utils::arrow::reverse(batch));
    MemTable::try_new(schema, vec![records])
}

#[derive(Debug, Clone)]
pub enum PartialTimeFilter {
    Low(Bound<NaiveDateTime>),
    High(Bound<NaiveDateTime>),
    Eq(NaiveDateTime),
}

impl PartialTimeFilter {
    fn try_from_expr(expr: &Expr, time_partition: &Option<String>) -> Option<Self> {
        let Expr::BinaryExpr(binexpr) = expr else {
            return None;
        };
        let (op, time) = extract_timestamp_bound(binexpr, time_partition)?;
        let value = match op {
            Operator::Gt => PartialTimeFilter::Low(Bound::Excluded(time)),
            Operator::GtEq => PartialTimeFilter::Low(Bound::Included(time)),
            Operator::Lt => PartialTimeFilter::High(Bound::Excluded(time)),
            Operator::LtEq => PartialTimeFilter::High(Bound::Included(time)),
            Operator::Eq => PartialTimeFilter::Eq(time),
            Operator::IsNotDistinctFrom => PartialTimeFilter::Eq(time),
            _ => return None,
        };
        Some(value)
    }

    pub fn binary_expr(&self, left: Expr) -> Expr {
        let (op, right) = match self {
            PartialTimeFilter::Low(Bound::Excluded(time)) => {
                (Operator::Gt, time.and_utc().timestamp_millis())
            }
            PartialTimeFilter::Low(Bound::Included(time)) => {
                (Operator::GtEq, time.and_utc().timestamp_millis())
            }
            PartialTimeFilter::High(Bound::Excluded(time)) => {
                (Operator::Lt, time.and_utc().timestamp_millis())
            }
            PartialTimeFilter::High(Bound::Included(time)) => {
                (Operator::LtEq, time.and_utc().timestamp_millis())
            }
            PartialTimeFilter::Eq(time) => (Operator::Eq, time.and_utc().timestamp_millis()),
            _ => unimplemented!(),
        };

        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            op,
            Box::new(Expr::Literal(
                ScalarValue::TimestampMillisecond(Some(right), None),
                None,
            )),
        ))
    }
}

fn is_overlapping_query(
    manifest_list: &[ManifestItem],
    time_filters: &[PartialTimeFilter],
) -> bool {
    // This is for backwards compatiblity. Older table format relies on listing.
    // if the start time is lower than lower bound of first file then we consider it overlapping
    let Some(first_entry_lower_bound) =
        manifest_list.iter().map(|file| file.time_lower_bound).min()
    else {
        return true;
    };

    for filter in time_filters {
        match filter {
            PartialTimeFilter::Low(Bound::Excluded(time))
            | PartialTimeFilter::Low(Bound::Included(time))
                if time < &first_entry_lower_bound.naive_utc() =>
            {
                return true;
            }
            _ => {}
        }
    }

    false
}

/// This function will accept time filters provided to the query and will split them
/// into listing time filters and manifest time filters
/// This makes parseable backwards compatible for when it did not have manifests
/// Logic-
/// The control flow will only come to this function if there exists data without manifest files
/// Two new time filter vec![] are created
/// For listing table time filters, we will use OG time filter low bound and either OG time filter upper bound
/// or manifest lower bound
/// For manifest time filter, we will manifest lower bound and OG upper bound
fn return_listing_time_filters(
    manifest_list: &[ManifestItem],
    time_filters: &mut Vec<PartialTimeFilter>,
) -> Option<Vec<PartialTimeFilter>> {
    if manifest_list.is_empty() {
        return Some(time_filters.clone());
    }

    // vec to hold timestamps for listing
    let mut vec_listing_timestamps = Vec::new();

    let mut first_entry_lower_bound = manifest_list
        .iter()
        .map(|file| file.time_lower_bound.naive_utc())
        .min()?;

    let mut new_time_filters = vec![PartialTimeFilter::Low(Bound::Included(
        first_entry_lower_bound,
    ))];

    time_filters.iter_mut().for_each(|filter| {
        match filter {
            // since we've already determined that there is a need to list tables,
            // we just need to check whether the filter's upper bound is < manifest lower bound
            PartialTimeFilter::High(Bound::Included(upper))
            | PartialTimeFilter::High(Bound::Excluded(upper)) => {
                if upper.lt(&&mut first_entry_lower_bound) {
                    // filter upper bound is less than manifest lower bound, continue using filter upper bound
                    vec_listing_timestamps.push(filter.clone());
                } else {
                    // use manifest lower bound as excluded
                    vec_listing_timestamps.push(PartialTimeFilter::High(Bound::Excluded(
                        first_entry_lower_bound,
                    )));
                }
                new_time_filters.push(filter.clone());
            }
            _ => {
                vec_listing_timestamps.push(filter.clone());
            }
        }
    });

    // update time_filters
    *time_filters = new_time_filters;

    if vec_listing_timestamps.len().gt(&0) {
        Some(vec_listing_timestamps)
    } else {
        None
    }
}

/// We should consider data in staging for queries concerning a time period,
/// ending within 5 minutes from now. e.g. If current time is 5
pub fn is_within_staging_window(time_filters: &[PartialTimeFilter]) -> bool {
    let five_minutes_back = (Utc::now() - TimeDelta::minutes(5))
        .with_second(0)
        .and_then(|x| x.with_nanosecond(0))
        .expect("zeroed value is valid")
        .naive_utc();

    if time_filters.iter().any(|filter| match filter {
        PartialTimeFilter::High(Bound::Excluded(time))
        | PartialTimeFilter::High(Bound::Included(time))
        | PartialTimeFilter::Eq(time) => time >= &five_minutes_back,
        _ => false,
    }) {
        return true;
    }

    // does it even have a higher bound
    let has_upper_bound = time_filters
        .iter()
        .any(|filter| matches!(filter, PartialTimeFilter::High(_)));

    !has_upper_bound
}

fn expr_in_boundary(filter: &Expr) -> bool {
    let Expr::BinaryExpr(binexpr) = filter else {
        return false;
    };
    let Some((op, time)) = extract_timestamp_bound(binexpr, &None) else {
        return false;
    };

    // this is due to knowlege of prefixes being minute long always.
    // Without a consistent partition spec this cannot be guarenteed.
    time.second() == 0
        && time.nanosecond() == 0
        && matches!(
            op,
            Operator::Gt | Operator::GtEq | Operator::Lt | Operator::LtEq
        )
}

fn extract_timestamp_bound(
    binexpr: &BinaryExpr,
    time_partition: &Option<String>,
) -> Option<(Operator, NaiveDateTime)> {
    let Expr::Literal(value, None) = binexpr.right.as_ref() else {
        return None;
    };

    let is_time_partition = match (binexpr.left.as_ref(), time_partition) {
        (Expr::Column(column), Some(time_partition)) => &column.name == time_partition,
        _ => false,
    };

    match value {
        ScalarValue::TimestampMillisecond(Some(value), _) => Some((
            binexpr.op,
            DateTime::from_timestamp_millis(*value).unwrap().naive_utc(),
        )),
        ScalarValue::TimestampNanosecond(Some(value), _) => Some((
            binexpr.op,
            DateTime::from_timestamp_nanos(*value).naive_utc(),
        )),
        ScalarValue::Utf8(Some(str_value)) if is_time_partition => {
            match str_value.parse::<NaiveDateTime>() {
                Ok(dt) => Some((binexpr.op, dt)),
                Err(_) => None,
            }
        }
        _ => None,
    }
}

// Extract start time and end time from filter predicate
pub fn extract_primary_filter(
    filters: &[Expr],
    time_partition: &Option<String>,
) -> Vec<PartialTimeFilter> {
    filters
        .iter()
        .filter_map(|expr| {
            let mut time_filter = None;
            let _ = expr.apply(&mut |expr| {
                if let Some(time) = PartialTimeFilter::try_from_expr(expr, time_partition) {
                    time_filter = Some(time);
                    Ok(TreeNodeRecursion::Stop) // Stop further traversal
                } else {
                    Ok(TreeNodeRecursion::Jump) // Skip this node
                }
            });
            time_filter
        })
        .collect()
}

pub trait ManifestExt: ManifestFile {
    fn find_matching_column(&self, partial_filter: &Expr) -> Option<&Column> {
        let name = match partial_filter {
            Expr::BinaryExpr(binary_expr) => filter_column_name(binary_expr.left.as_ref())?,
            _ => {
                return None;
            }
        };

        self.columns().iter().find(|col| &col.name == name)
    }

    fn can_be_pruned(&self, partial_filter: &Expr) -> bool {
        match partial_filter {
            Expr::BinaryExpr(binary_expr) if binary_expr.op == Operator::And => {
                return self.can_be_pruned(binary_expr.left.as_ref())
                    || self.can_be_pruned(binary_expr.right.as_ref());
            }
            Expr::BinaryExpr(binary_expr) if binary_expr.op == Operator::Or => {
                return self.can_be_pruned(binary_expr.left.as_ref())
                    && self.can_be_pruned(binary_expr.right.as_ref());
            }
            Expr::Between(between) if !between.negated => {
                if let Some(can_prune) = can_prune_between(self.columns(), between) {
                    return can_prune;
                }
                let low = Expr::BinaryExpr(BinaryExpr::new(
                    between.expr.clone(),
                    Operator::GtEq,
                    between.low.clone(),
                ));
                let high = Expr::BinaryExpr(BinaryExpr::new(
                    between.expr.clone(),
                    Operator::LtEq,
                    between.high.clone(),
                ));
                return self.can_be_pruned(&low) || self.can_be_pruned(&high);
            }
            _ => {}
        }

        fn extract_op_scalar(expr: &Expr) -> Option<(Operator, &ScalarValue)> {
            let Expr::BinaryExpr(expr) = expr else {
                return None;
            };
            let Expr::Literal(value, None) = &*expr.right else {
                return None;
            };
            /* `BinaryExp` doesn't implement `Copy` */
            Some((expr.op, value))
        }

        if let Some((column_name, pattern, escape_char)) = extract_like_pattern(partial_filter)
            && let Some(col) = self.columns().iter().find(|col| col.name == column_name)
        {
            let grams = like_literal_index_terms(pattern, escape_char);
            if let Some(text_ngrams) = &col.text_ngrams
                && text_ngrams.complete
                && grams
                    .iter()
                    .filter(|gram| gram.chars().count() >= text_ngrams.min_len)
                    .any(|gram| !text_ngrams.contains(gram))
            {
                return true;
            }
            if let Some(text_ngram_hashes) = &col.text_ngram_hashes
                && text_ngram_hashes.complete
                && grams
                    .iter()
                    .filter(|gram| gram.chars().count() >= text_ngram_hashes.min_len)
                    .any(|gram| !text_ngram_hashes.contains(gram))
            {
                return true;
            }
        }

        let Some(col) = self.find_matching_column(partial_filter) else {
            return false;
        };

        let Some((op, value)) = extract_op_scalar(partial_filter) else {
            return false;
        };

        let Some(value) = cast_or_none(value) else {
            return false;
        };

        if let Some(exact_values) = &col.exact_values
            && exact_values.complete
        {
            if matches!(op, Operator::Eq | Operator::IsNotDistinctFrom)
                && !exact_values.contains(&value.exact_index_key())
            {
                return true;
            }

            if let Some(any_match) =
                exact_values_may_satisfy(exact_values, value, op, col.stats.as_ref())
                && !any_match
            {
                return true;
            }
        }
        if let Some(exact_hashes) = &col.exact_hashes
            && exact_hashes.complete
            && matches!(op, Operator::Eq | Operator::IsNotDistinctFrom)
            && !exact_hashes.contains_value(&value.exact_index_key())
        {
            return true;
        }

        let Some(stats) = &col.stats else {
            return false;
        };

        !satisfy_constraints(value, op, stats).unwrap_or(true)
    }
}

impl<T: ManifestFile> ManifestExt for T {}

fn filter_column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(col) => Some(&col.name),
        Expr::Cast(cast) => filter_column_name(cast.expr.as_ref()),
        Expr::TryCast(cast) => filter_column_name(cast.expr.as_ref()),
        _ => None,
    }
}

fn can_prune_between(columns: &[Column], between: &Between) -> Option<bool> {
    let column_name = filter_column_name(between.expr.as_ref())?;
    let col = columns.iter().find(|col| col.name == column_name)?;
    let low = literal_cast_or_none(between.low.as_ref())?;
    let high = literal_cast_or_none(between.high.as_ref())?;

    if let Some(exact_values) = &col.exact_values
        && exact_values.complete
        && let Some(any_match) =
            exact_values_between_may_satisfy(exact_values, low, high, col.stats.as_ref())
    {
        return Some(!any_match);
    }

    None
}

fn literal_cast_or_none(expr: &Expr) -> Option<CastRes<'_>> {
    let Expr::Literal(value, None) = expr else {
        return None;
    };
    cast_or_none(value)
}

fn parquet_scan_batch_size(configured: usize, limit: Option<usize>) -> usize {
    let configured = configured.max(1);
    match limit {
        Some(limit) => configured.min(limit.max(1024)),
        None => configured,
    }
}

fn extract_like_pattern(expr: &Expr) -> Option<(&str, &str, Option<char>)> {
    let Expr::Like(like) = expr else {
        return None;
    };
    if like.negated {
        return None;
    }

    let Expr::Column(col) = like.expr.as_ref() else {
        return None;
    };
    let pattern = match like.pattern.as_ref() {
        Expr::Literal(ScalarValue::Utf8(Some(pattern)), None)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(pattern)), None)
        | Expr::Literal(ScalarValue::Utf8View(Some(pattern)), None) => pattern,
        _ => {
            return None;
        }
    };

    Some((&col.name, pattern, like.escape_char))
}

fn like_literal_index_terms(pattern: &str, escape_char: Option<char>) -> Vec<String> {
    let mut grams = Vec::new();
    let mut literal = String::new();
    let mut escaped = false;

    for ch in pattern.chars() {
        if escape_char == Some(ch) && !escaped {
            escaped = true;
            continue;
        }

        if !escaped && matches!(ch, '%' | '_') {
            push_literal_index_terms(&literal, &mut grams);
            literal.clear();
            continue;
        }

        literal.push(ch);
        escaped = false;
    }
    push_literal_index_terms(&literal, &mut grams);
    grams.sort();
    grams.dedup();
    grams
}

fn push_literal_index_terms(literal: &str, grams: &mut Vec<String>) {
    let chars = literal.to_lowercase().chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return;
    }

    let width = chars.len().min(3);
    grams.extend(chars.windows(width).map(|window| window.iter().collect()));
}

#[derive(Clone, Copy)]
enum CastRes<'a> {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(&'a str),
}

impl CastRes<'_> {
    fn exact_index_key(self) -> String {
        match self {
            Self::Bool(value) => value.to_string(),
            Self::Int(value) => value.to_string(),
            Self::Float(value) => value.to_string(),
            Self::String(value) => value.to_string(),
        }
    }
}

fn cast_or_none(scalar: &ScalarValue) -> Option<CastRes<'_>> {
    match scalar {
        ScalarValue::Null => None,
        ScalarValue::Boolean(val) => val.map(CastRes::Bool),
        ScalarValue::Float32(val) => val.map(|val| CastRes::Float(val as f64)),
        ScalarValue::Float64(val) => val.map(CastRes::Float),
        ScalarValue::Int8(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::Int16(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::Int32(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::Int64(val) => val.map(CastRes::Int),
        ScalarValue::UInt8(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::UInt16(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::UInt32(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::UInt64(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::Utf8(val) => val.as_ref().map(|val| CastRes::String(val)),
        ScalarValue::Date32(val) => val.map(|val| CastRes::Int(val as i64)),
        ScalarValue::TimestampMillisecond(val, _) => val.map(CastRes::Int),
        _ => None,
    }
}

fn satisfy_constraints(value: CastRes, op: Operator, stats: &TypedStatistics) -> Option<bool> {
    fn matches<T: std::cmp::PartialOrd>(value: T, min: T, max: T, op: Operator) -> Option<bool> {
        let val = match op {
            Operator::Eq | Operator::IsNotDistinctFrom => value >= min && value <= max,
            Operator::Lt => value > min,
            Operator::LtEq => value >= min,
            Operator::Gt => value < max,
            Operator::GtEq => value <= max,
            _ => return None,
        };
        Some(val)
    }

    match (value, stats) {
        (CastRes::Bool(val), TypedStatistics::Bool(stats)) => {
            matches(val, stats.min, stats.max, op)
        }
        (CastRes::Int(val), TypedStatistics::Int(stats)) => matches(val, stats.min, stats.max, op),
        (CastRes::Float(val), TypedStatistics::Float(stats)) => {
            matches(val, stats.min, stats.max, op)
        }
        (CastRes::String(val), TypedStatistics::String(stats)) => {
            matches(val, &stats.min, &stats.max, op)
        }
        _ => None,
    }
}

fn exact_values_may_satisfy(
    exact_values: &crate::catalog::column::ExactValues,
    value: CastRes<'_>,
    op: Operator,
    stats: Option<&TypedStatistics>,
) -> Option<bool> {
    match (value, stats?) {
        (CastRes::Bool(query), TypedStatistics::Bool(_)) => exact_values_any_parse_match(
            &exact_values.values,
            query,
            op,
            |raw| raw.parse::<bool>().ok(),
            |_| true,
        ),
        (CastRes::Int(query), TypedStatistics::Int(_)) => exact_values_any_parse_match(
            &exact_values.values,
            query,
            op,
            |raw| raw.parse::<i64>().ok(),
            |_| true,
        ),
        (CastRes::Float(query), TypedStatistics::Float(_)) if !query.is_nan() => {
            exact_values_any_parse_match(
                &exact_values.values,
                query,
                op,
                |raw| raw.parse::<f64>().ok(),
                |candidate| !candidate.is_nan(),
            )
        }
        (CastRes::String(query), TypedStatistics::String(_)) => Some(
            exact_values
                .values
                .iter()
                .any(|candidate| compare_exact(candidate.as_str(), query, op)),
        ),
        _ => None,
    }
}

fn exact_values_between_may_satisfy(
    exact_values: &crate::catalog::column::ExactValues,
    low: CastRes<'_>,
    high: CastRes<'_>,
    stats: Option<&TypedStatistics>,
) -> Option<bool> {
    match (low, high, stats?) {
        (CastRes::Bool(low), CastRes::Bool(high), TypedStatistics::Bool(_)) => {
            exact_values_any_parse_between_match(
                &exact_values.values,
                low,
                high,
                |raw| raw.parse::<bool>().ok(),
                |_| true,
            )
        }
        (CastRes::Int(low), CastRes::Int(high), TypedStatistics::Int(_)) => {
            exact_values_any_parse_between_match(
                &exact_values.values,
                low,
                high,
                |raw| raw.parse::<i64>().ok(),
                |_| true,
            )
        }
        (CastRes::Float(low), CastRes::Float(high), TypedStatistics::Float(_))
            if !low.is_nan() && !high.is_nan() =>
        {
            exact_values_any_parse_between_match(
                &exact_values.values,
                low,
                high,
                |raw| raw.parse::<f64>().ok(),
                |candidate| !candidate.is_nan(),
            )
        }
        (CastRes::String(low), CastRes::String(high), TypedStatistics::String(_)) => {
            if low > high {
                return Some(false);
            }
            Some(
                exact_values
                    .values
                    .iter()
                    .any(|candidate| candidate.as_str() >= low && candidate.as_str() <= high),
            )
        }
        _ => None,
    }
}

fn exact_values_any_parse_match<T, F, V>(
    values: &[String],
    query: T,
    op: Operator,
    parse: F,
    valid: V,
) -> Option<bool>
where
    T: Copy + PartialOrd + PartialEq,
    F: Fn(&str) -> Option<T>,
    V: Fn(T) -> bool,
{
    let mut any_match = false;
    for raw in values {
        let candidate = parse(raw)?;
        if !valid(candidate) {
            return None;
        }
        any_match |= compare_exact(candidate, query, op);
    }
    Some(any_match)
}

fn exact_values_any_parse_between_match<T, F, V>(
    values: &[String],
    low: T,
    high: T,
    parse: F,
    valid: V,
) -> Option<bool>
where
    T: Copy + PartialOrd,
    F: Fn(&str) -> Option<T>,
    V: Fn(T) -> bool,
{
    if low > high {
        return Some(false);
    }

    for raw in values {
        let candidate = parse(raw)?;
        if !valid(candidate) {
            return None;
        }
        if candidate >= low && candidate <= high {
            return Some(true);
        }
    }
    Some(false)
}

fn compare_exact<T: PartialOrd + PartialEq>(candidate: T, query: T, op: Operator) -> bool {
    match op {
        Operator::Eq | Operator::IsNotDistinctFrom => candidate == query,
        Operator::Lt => candidate < query,
        Operator::LtEq => candidate <= query,
        Operator::Gt => candidate > query,
        Operator::GtEq => candidate >= query,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Add;

    use arrow_schema::DataType;
    use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime, NaiveTime, Utc};
    use datafusion::{
        logical_expr::{Between, BinaryExpr, Operator},
        prelude::Expr,
        scalar::ScalarValue,
    };

    use crate::catalog::{
        column::{
            Column, ExactHashes, ExactValues, Int64Type, TextNgramHashes, TextNgrams,
            TypedStatistics, Utf8Type, exact_value_hash, text_ngram_hash,
        },
        manifest::File,
        snapshot::ManifestItem,
    };

    use super::{
        ManifestExt, PartialTimeFilter, extract_timestamp_bound, is_overlapping_query,
        like_literal_index_terms, parquet_scan_batch_size,
    };

    fn datetime_min(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_time(NaiveTime::MIN)
            .and_utc()
    }

    fn datetime_max(year: i32, month: u32, day: u32) -> DateTime<Utc> {
        NaiveDate::from_ymd_opt(year, month, day)
            .unwrap()
            .and_hms_milli_opt(23, 59, 59, 999)
            .unwrap()
            .and_utc()
    }

    fn manifest_items() -> Vec<ManifestItem> {
        vec![
            ManifestItem {
                manifest_path: "1".to_string(),
                time_lower_bound: datetime_min(2023, 12, 15),
                time_upper_bound: datetime_max(2023, 12, 15),
                events_ingested: 0,
                ingestion_size: 0,
                storage_size: 0,
            },
            ManifestItem {
                manifest_path: "2".to_string(),
                time_lower_bound: datetime_min(2023, 12, 16),
                time_upper_bound: datetime_max(2023, 12, 16),
                events_ingested: 0,
                ingestion_size: 0,
                storage_size: 0,
            },
            ManifestItem {
                manifest_path: "3".to_string(),
                time_lower_bound: datetime_min(2023, 12, 17),
                time_upper_bound: datetime_max(2023, 12, 17),
                events_ingested: 0,
                ingestion_size: 0,
                storage_size: 0,
            },
        ]
    }

    fn exact_index_filter(value: &str) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(Expr::Column("trace_id".into())),
            Operator::Eq,
            Box::new(Expr::Literal(
                ScalarValue::Utf8(Some(value.to_string())),
                None,
            )),
        ))
    }

    fn exact_index_file(values: &[&str], complete: bool) -> File {
        let mut values = values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        values.sort();

        File {
            file_path: "file.parquet".to_string(),
            num_rows: 10,
            file_size: 10,
            ingestion_size: 10,
            columns: vec![Column {
                name: "trace_id".to_string(),
                stats: Some(TypedStatistics::String(Utf8Type {
                    min: "a".to_string(),
                    max: "z".to_string(),
                })),
                exact_values: Some(ExactValues { complete, values }),
                exact_hashes: None,
                text_ngrams: None,
                text_ngram_hashes: None,
                uncompressed_size: 10,
                compressed_size: 10,
            }],
            sort_order_id: Vec::new(),
        }
    }

    fn exact_hash_index_file(values: &[&str], complete: bool) -> File {
        let mut hashes = values
            .iter()
            .map(|value| exact_value_hash(value))
            .collect::<Vec<_>>();
        hashes.sort();

        File {
            file_path: "file.parquet".to_string(),
            num_rows: 10,
            file_size: 10,
            ingestion_size: 10,
            columns: vec![Column {
                name: "trace_id".to_string(),
                stats: Some(TypedStatistics::String(Utf8Type {
                    min: "a".to_string(),
                    max: "z".to_string(),
                })),
                exact_values: None,
                exact_hashes: Some(ExactHashes { complete, hashes }),
                text_ngrams: None,
                text_ngram_hashes: None,
                uncompressed_size: 10,
                compressed_size: 10,
            }],
            sort_order_id: Vec::new(),
        }
    }

    fn like_filter(pattern: &str) -> Expr {
        Expr::Like(datafusion::logical_expr::Like::new(
            false,
            Box::new(Expr::Column("body".into())),
            Box::new(Expr::Literal(
                ScalarValue::Utf8(Some(pattern.to_string())),
                None,
            )),
            None,
            false,
        ))
    }

    fn text_ngram_file(grams: &[&str], complete: bool) -> File {
        text_ngram_file_with_min_len(grams, complete, 1)
    }

    fn text_ngram_file_with_min_len(grams: &[&str], complete: bool, min_len: usize) -> File {
        let mut grams = grams
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        grams.sort();

        File {
            file_path: "file.parquet".to_string(),
            num_rows: 10,
            file_size: 10,
            ingestion_size: 10,
            columns: vec![Column {
                name: "body".to_string(),
                stats: None,
                exact_values: None,
                exact_hashes: None,
                text_ngrams: Some(TextNgrams {
                    complete,
                    min_len,
                    grams,
                }),
                text_ngram_hashes: None,
                uncompressed_size: 10,
                compressed_size: 10,
            }],
            sort_order_id: Vec::new(),
        }
    }

    fn text_ngram_hash_file(grams: &[&str], complete: bool) -> File {
        let mut hashes = grams
            .iter()
            .map(|value| text_ngram_hash(value))
            .collect::<Vec<_>>();
        hashes.sort();

        File {
            file_path: "file.parquet".to_string(),
            num_rows: 10,
            file_size: 10,
            ingestion_size: 10,
            columns: vec![Column {
                name: "body".to_string(),
                stats: None,
                exact_values: None,
                exact_hashes: None,
                text_ngrams: None,
                text_ngram_hashes: Some(TextNgramHashes {
                    complete,
                    min_len: 1,
                    hashes,
                }),
                uncompressed_size: 10,
                compressed_size: 10,
            }],
            sort_order_id: Vec::new(),
        }
    }

    fn numeric_stats_file(min: i64, max: i64) -> File {
        File {
            file_path: "file.parquet".to_string(),
            num_rows: 10,
            file_size: 10,
            ingestion_size: 10,
            columns: vec![Column {
                name: "duration_ms".to_string(),
                stats: Some(TypedStatistics::Int(Int64Type { min, max })),
                exact_values: None,
                exact_hashes: None,
                text_ngrams: None,
                text_ngram_hashes: None,
                uncompressed_size: 10,
                compressed_size: 10,
            }],
            sort_order_id: Vec::new(),
        }
    }

    fn numeric_exact_index_file(values: &[&str], min: i64, max: i64) -> File {
        let mut values = values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>();
        values.sort();

        File {
            file_path: "file.parquet".to_string(),
            num_rows: 10,
            file_size: 10,
            ingestion_size: 10,
            columns: vec![Column {
                name: "duration_ms".to_string(),
                stats: Some(TypedStatistics::Int(Int64Type { min, max })),
                exact_values: Some(ExactValues {
                    complete: true,
                    values,
                }),
                exact_hashes: None,
                text_ngrams: None,
                text_ngram_hashes: None,
                uncompressed_size: 10,
                compressed_size: 10,
            }],
            sort_order_id: Vec::new(),
        }
    }

    fn cast_numeric_filter(value: i64, op: Operator) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(Expr::Cast(datafusion::logical_expr::Cast::new(
                Box::new(Expr::Column("duration_ms".into())),
                DataType::Int64,
            ))),
            op,
            Box::new(Expr::Literal(ScalarValue::Int64(Some(value)), None)),
        ))
    }

    fn numeric_between_filter(low: i64, high: i64) -> Expr {
        Expr::Between(Between::new(
            Box::new(Expr::Cast(datafusion::logical_expr::Cast::new(
                Box::new(Expr::Column("duration_ms".into())),
                DataType::Int64,
            ))),
            false,
            Box::new(Expr::Literal(ScalarValue::Int64(Some(low)), None)),
            Box::new(Expr::Literal(ScalarValue::Int64(Some(high)), None)),
        ))
    }

    fn and_filter(left: Expr, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            Operator::And,
            Box::new(right),
        ))
    }

    fn or_filter(left: Expr, right: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr::new(
            Box::new(left),
            Operator::Or,
            Box::new(right),
        ))
    }

    #[test]
    fn exact_index_prunes_missing_equality_inside_min_max_range() {
        let file = exact_index_file(&["a", "z"], true);

        assert!(file.can_be_pruned(&exact_index_filter("m")));
        assert!(!file.can_be_pruned(&exact_index_filter("a")));
    }

    #[test]
    fn incomplete_exact_index_does_not_prune() {
        let file = exact_index_file(&["a", "z"], false);

        assert!(!file.can_be_pruned(&exact_index_filter("m")));
    }

    #[test]
    fn exact_hash_index_prunes_missing_equality_inside_min_max_range() {
        let file = exact_hash_index_file(&["a", "z"], true);

        assert!(file.can_be_pruned(&exact_index_filter("m")));
        assert!(!file.can_be_pruned(&exact_index_filter("a")));
    }

    #[test]
    fn incomplete_exact_hash_index_does_not_prune() {
        let file = exact_hash_index_file(&["a", "z"], false);

        assert!(!file.can_be_pruned(&exact_index_filter("m")));
    }

    #[test]
    fn text_ngram_index_prunes_like_when_required_trigram_missing() {
        let file = text_ngram_file(&["ups", "pst", "str", "tre", "rea", "eam"], true);

        assert!(file.can_be_pruned(&like_filter("%definitely_not_present_token%")));
        assert!(!file.can_be_pruned(&like_filter("%upstream%")));
    }

    #[test]
    fn incomplete_text_ngram_index_does_not_prune_like() {
        let file = text_ngram_file(&["ups"], false);

        assert!(!file.can_be_pruned(&like_filter("%definitely_not_present_token%")));
    }

    #[test]
    fn text_ngram_hash_index_prunes_like_when_required_term_missing() {
        let file = text_ngram_hash_file(&["ups", "pst", "str", "tre", "rea", "eam"], true);

        assert!(file.can_be_pruned(&like_filter("%definitely_not_present_token%")));
        assert!(!file.can_be_pruned(&like_filter("%upstream%")));
    }

    #[test]
    fn incomplete_text_ngram_hash_index_does_not_prune_like() {
        let file = text_ngram_hash_file(&["ups"], false);

        assert!(!file.can_be_pruned(&like_filter("%definitely_not_present_token%")));
    }

    #[test]
    fn text_ngram_index_prunes_short_like_when_required_bigram_missing() {
        let file = text_ngram_file(&["q", "x", "x=", "=1"], true);

        assert!(file.can_be_pruned(&like_filter("%q1%")));
        assert!(!file.can_be_pruned(&like_filter("%x=%")));
    }

    #[test]
    fn legacy_trigram_index_does_not_prune_short_like() {
        let file = text_ngram_file_with_min_len(&["ups"], true, 3);

        assert!(!file.can_be_pruned(&like_filter("%q1%")));
    }

    #[test]
    fn like_literal_index_terms_respects_wildcards_and_escapes() {
        assert_eq!(like_literal_index_terms("%cart_id%", None)[0], "art");
        assert_eq!(
            like_literal_index_terms("%ab%", None),
            vec!["ab".to_string()]
        );
        assert!(like_literal_index_terms(r"%abc\_def%", Some('\\')).contains(&"c_d".to_string()));
        assert!(like_literal_index_terms("%/123", None).contains(&"123".to_string()));
        assert!(like_literal_index_terms("%value with spaces%", None).contains(&" wi".to_string()));
        assert!(
            like_literal_index_terms("%cache_key=product:%", None).contains(&"y=p".to_string())
        );
    }

    #[test]
    fn parquet_batch_size_uses_smaller_limit_aware_batches() {
        assert_eq!(parquet_scan_batch_size(20_000, Some(100)), 1024);
        assert_eq!(parquet_scan_batch_size(20_000, Some(4096)), 4096);
        assert_eq!(parquet_scan_batch_size(512, Some(100)), 512);
        assert_eq!(parquet_scan_batch_size(20_000, None), 20_000);
        assert_eq!(parquet_scan_batch_size(0, Some(100)), 1);
    }

    #[test]
    fn manifest_pruning_sees_column_through_cast() {
        let file = numeric_stats_file(100, 400);

        assert!(file.can_be_pruned(&cast_numeric_filter(1500, Operator::GtEq)));
        assert!(!file.can_be_pruned(&cast_numeric_filter(300, Operator::GtEq)));
    }

    #[test]
    fn complete_exact_index_prunes_numeric_range_inside_broad_min_max() {
        let file = numeric_exact_index_file(&["100", "200", "300"], 100, 10_000);

        assert!(file.can_be_pruned(&cast_numeric_filter(1500, Operator::GtEq)));
        assert!(!file.can_be_pruned(&cast_numeric_filter(200, Operator::GtEq)));
    }

    #[test]
    fn invalid_numeric_exact_index_value_does_not_prune_range() {
        let file = numeric_exact_index_file(&["100", "not-a-number"], 100, 10_000);

        assert!(!file.can_be_pruned(&cast_numeric_filter(1500, Operator::GtEq)));
    }

    #[test]
    fn and_filter_prunes_when_any_branch_cannot_match() {
        let file = exact_index_file(&["a"], true);

        let filter = and_filter(exact_index_filter("missing"), like_filter("%not_indexed%"));

        assert!(file.can_be_pruned(&filter));
    }

    #[test]
    fn or_filter_prunes_only_when_all_branches_cannot_match() {
        let file = exact_index_file(&["a"], true);

        assert!(file.can_be_pruned(&or_filter(
            exact_index_filter("missing"),
            exact_index_filter("also_missing")
        )));
        assert!(!file.can_be_pruned(&or_filter(
            exact_index_filter("missing"),
            exact_index_filter("a")
        )));
    }

    #[test]
    fn between_filter_uses_exact_numeric_index_for_pruning() {
        let file = numeric_exact_index_file(&["100", "200", "900"], 0, 10_000);

        assert!(file.can_be_pruned(&numeric_between_filter(300, 400)));
        assert!(!file.can_be_pruned(&numeric_between_filter(150, 250)));
    }

    #[test]
    fn bound_min_is_overlapping() {
        let res = is_overlapping_query(
            &manifest_items(),
            &[PartialTimeFilter::Low(std::ops::Bound::Included(
                datetime_min(2023, 12, 14).naive_utc(),
            ))],
        );

        assert!(res)
    }

    #[test]
    fn bound_min_plus_hour_is_overlapping() {
        let res = is_overlapping_query(
            &manifest_items(),
            &[PartialTimeFilter::Low(std::ops::Bound::Included(
                datetime_min(2023, 12, 14)
                    .naive_utc()
                    .add(Duration::hours(3)),
            ))],
        );

        assert!(res)
    }

    #[test]
    fn bound_next_day_min_is_not_overlapping() {
        let res = is_overlapping_query(
            &manifest_items(),
            &[PartialTimeFilter::Low(std::ops::Bound::Included(
                datetime_min(2023, 12, 16).naive_utc(),
            ))],
        );

        assert!(!res)
    }

    #[test]
    fn timestamp_in_milliseconds() {
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::TimestampMillisecond(Some(1672531200000), None),
                None,
            )),
        };

        let time_partition = Some("timestamp_column".to_string());
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        let expected = Some((
            Operator::Eq,
            NaiveDateTime::parse_from_str("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ));

        assert_eq!(result, expected);
    }

    #[test]
    fn timestamp_in_nanoseconds() {
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Gt,
            right: Box::new(Expr::Literal(
                ScalarValue::TimestampNanosecond(Some(1672531200000000000), None),
                None,
            )),
        };

        let time_partition = Some("timestamp_column".to_string());
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        let expected = Some((
            Operator::Gt,
            NaiveDateTime::parse_from_str("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap(),
        ));

        assert_eq!(result, expected);
    }

    #[test]
    fn string_timestamp() {
        let timestamp = "2023-01-01T00:00:00";
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Lt,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some(timestamp.to_owned())),
                None,
            )),
        };

        let time_partition = Some("timestamp_column".to_string());
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        let expected = Some((
            Operator::Lt,
            NaiveDateTime::parse_from_str(timestamp, "%Y-%m-%dT%H:%M:%S").unwrap(),
        ));

        assert_eq!(result, expected);
    }

    #[test]
    fn unexpected_utf8_column() {
        let timestamp = "2023-01-01T00:00:00";
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("other_column".into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::Utf8(Some(timestamp.to_owned())),
                None,
            )),
        };

        let time_partition = Some("timestamp_column".to_string());
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        assert!(result.is_none());
    }

    #[test]
    fn unsupported_literal_type() {
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(ScalarValue::Int32(Some(42)), None)),
        };

        let time_partition = Some("timestamp_column".to_string());
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        assert!(result.is_none());
    }

    #[test]
    fn no_literal_on_right() {
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Eq,
            right: Box::new(Expr::Column("other_column".into())),
        };

        let time_partition = Some("timestamp_column".to_string());
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        assert!(result.is_none());
    }

    #[test]
    fn non_time_partition_timestamps() {
        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::TimestampMillisecond(Some(1672531200000), None),
                None,
            )),
        };

        let time_partition = None;
        let result = extract_timestamp_bound(&binexpr, &time_partition);
        let expected = Some((
            Operator::Eq,
            NaiveDateTime::parse_from_str("2023-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S").unwrap(),
        ));

        assert_eq!(result, expected);

        let binexpr = BinaryExpr {
            left: Box::new(Expr::Column("timestamp_column".into())),
            op: Operator::Eq,
            right: Box::new(Expr::Literal(
                ScalarValue::TimestampNanosecond(Some(1672531200000000000), None),
                None,
            )),
        };
        let result = extract_timestamp_bound(&binexpr, &time_partition);

        assert_eq!(result, expected);
    }
}
