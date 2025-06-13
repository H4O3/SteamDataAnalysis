package org.H2O2;// E:\JavaProject\SteamDataAnalysis\DataCount\src\main\java\org\H2O2\DataProcessing.java

import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class DataProcessing {
    public static void main(String[] args) {
        // 初始化 Spark 会话
        // 创建 SparkSession 实例并设置应用程序名称、运行模式等
        SparkSession spark = SparkSession.builder()
                .appName("Audience Preference Model") // 设置应用名称
                .master("local[*]") // 在本地运行，并利用所有核心
                .getOrCreate(); // 创建或获取现有的 SparkSession

        // 读取 CSV 数据集
        // 使用 SparkSession 读取 steam.csv 文件，自动推断 schema 和包含 header
        Dataset<Row> steamDF = spark.read()
                .option("header", true) // CSV 文件有表头
                .option("inferSchema", true) // 自动推断数据类型
                .csv("steam.csv"); // 数据源路径

        // 数据诊断
        steamDF.describe().show();
        System.out.println("缺失值统计:");
        for (String column : steamDF.columns()) {
            long nullCount = steamDF.filter(col(column).isNull()).count();
            System.out.println(column + ": " + nullCount);
        }

        // 数据清洗
        // 1. 移除重复的行，确保数据的唯一性
        // 2. 使用0填充缺失值，保证数据的完整性
        // 3. 将相关列转换为整型，提高数据的一致性和可处理性
        // 4. 过滤掉价格为空或非正数的记录，确保数据的有效性和准确性
        Dataset<Row> cleanedDF = steamDF
                .dropDuplicates()
                .na().drop()
                .withColumn("positive_ratings", col("positive_ratings").cast("int"))
                .withColumn("negative_ratings", col("negative_ratings").cast("int"))
                .withColumn("average_playtime", col("average_playtime").cast("double"))
                .withColumn("median_playtime", col("median_playtime").cast("double"))
                .filter(col("price").isNotNull().and(col("price").gt(0)));


        // 数据变换 - 创建新特征
        // 在cleanedDF数据集的基础上，进行数据转换以添加新的列
        Dataset<Row> transformedDF = cleanedDF
                // 新增"total_ratings"列，表示正面评价数量加上负面评价数量的总评价数
                .withColumn("total_ratings", col("positive_ratings").plus(col("negative_ratings")))
                // 新增"rating_ratio"列，表示正面评价占总评价数的比例
                .withColumn("rating_ratio", col("positive_ratings").divide(col("total_ratings")))
                // 创建列 rating_category：
                // 根据好评率将游戏分为六个等级：好评如潮、特别好评、多半好评、褒贬不一、多半差评、差评如潮
                .withColumn("rating_category",
                        when(col("rating_ratio").geq(0.95), lit("好评如潮"))
                                .when(col("rating_ratio").geq(0.8).and(col("rating_ratio").lt(0.95)), lit("特别好评"))
                                .when(col("rating_ratio").geq(0.7).and(col("rating_ratio").lt(0.8)), lit("多半好评"))
                                .when(col("rating_ratio").geq(0.4).and(col("rating_ratio").lt(0.7)), lit("褒贬不一"))
                                .when(col("rating_ratio").geq(0.2).and(col("rating_ratio").lt(0.4)), lit("多半差评"))
                                .otherwise(lit("差评如潮")))
                // 创建列 label：
                // 如果 negative_ratings 为 0，则 label 为 0；
                // 如果 positive_ratings / negative_ratings > 10，则 label 为 1；
                // 其他情况 label 为 0。
                // 1表示受欢迎
                .withColumn("label",
                        when(col("negative_ratings").equalTo(0), 0)
                                .when(col("positive_ratings").divide(col("negative_ratings")).gt(10), 1)
                                .otherwise(0)
                );



        spark.stop();
    }
}
