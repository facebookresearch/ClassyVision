#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import sys

# NOTE: Edit /usr/local/jdk-8u60-64/jre/lib/logging.properties to hide all logs
import metastore
from hiveio import par_init
from hiveio_cpp import hiveio


# set classpath:
par_init.install_class_path()


def download_from_hive(
    namespace,
    tablename,
    everstore_column,
    label_column,
    partition_column,
    partition_column_values,
):
    """
    Function to load IDs from Hive.
    """
    # disable HiveIO info logs
    os.environ["GLOG_minloglevel"] = "3"
    # set partition name and columns:
    partitions = [
        "{0}={1}".format(partition_column, partition_column_value)
        for partition_column_value in partition_column_values
    ]
    columns = [everstore_column, label_column]

    # check metastore for number of rows:
    sys.argv = ["."]
    ms = metastore.metastore(namespace=namespace)
    num_rows = 0
    for partition in partitions:
        assert ms.exists_partition(
            tablename, partition
        ), "partition not found: {0}/{1}".format(tablename, partition)

        p = ms.get_partition(tablename, partition)
        num_rows += int(p.parameters["numRows"])

    # start HiveIO reader:
    batch_size = 10000
    hiveio.start_reading(
        namespace=namespace,
        table=tablename,
        partitions=partitions,
        column_names=columns,
        batch_size=batch_size,
        max_queued_batches=10,
    )

    # read all data:
    handles, labels, cnt = [None] * num_rows, [None] * num_rows, 0
    while True:
        # read and process batch:
        batch = hiveio.get_batch()
        if batch == [] or cnt + len(batch) > num_rows:
            break
        cur_handles = [val[0] for val in batch]
        cur_labels = [val[1] for val in batch]

        # store data:
        handles[cnt : cnt + len(batch)] = cur_handles
        labels[cnt : cnt + len(batch)] = cur_labels
        cnt += len(batch)
        logging.info("Downloaded {0} of {1} rows.".format(cnt, num_rows))

    # close reader and return data:
    hiveio.stop_reading()
    if cnt < num_rows:
        handles = handles[0:cnt]
        labels = labels[0:cnt]
    return handles, labels


def get_partition_to_num_rows(
    namespace, tablename, partition_column, partition_column_values
):
    """
    Helper function to get total num_rows in hive for given
    partition_column_values.
    """
    partitions = {
        "{0}={1}".format(partition_column, partition_column_value)
        for partition_column_value in partition_column_values
    }
    # Setting higher number of retries, as during testing, sometimes default
    # "retries" values didn't seem enough in some cases.
    ms = metastore.metastore(
        namespace=namespace,
        meta_only=True,
        retries=10,
        # timeout in milliseconds.
        timeout=1800000,
    )
    partition_to_num_rows = {}

    all_partitions = ms.get_partitions(tablename)
    for hive_partition in all_partitions:
        assert "numRows" in hive_partition.parameters, (
            "numRows not in hive_partition.parameters,"
            "Do not use Presto tables, only Hive tables!')"
        )
        if hive_partition.partitionName in partitions:
            patition_column_value = hive_partition.partitionName.split("=")[1]
            partition_to_num_rows[patition_column_value] = int(
                hive_partition.parameters["numRows"]
            )

    return partition_to_num_rows
