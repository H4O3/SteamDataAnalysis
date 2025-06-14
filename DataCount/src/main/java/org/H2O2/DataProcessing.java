package org.H2O2;

import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

public class DataProcessing {
    public static void main(String[] args) {
        // 1. 初始化Spark会话
        SparkSession spark = SparkSession.builder()
                .appName("Audience Preference Model") // 应用名称
                .master("local[*]")                  // 本地模式使用所有CPU核心
                .getOrCreate();

        // 2. 读取CSV数据源
        Dataset<Row> steamDF = spark.read()
                .option("header", true)     // 首行为列名
                .option("inferSchema", true) // 自动推断数据类型
                .csv("steam.csv");          // 数据文件路径

        // 3. 数据探索
        steamDF.describe().show();  // 显示数值型列的统计摘要
        System.out.println("缺失值统计:");
        for (String column : steamDF.columns()) {
            long nullCount = steamDF.filter(col(column).isNull()).count(); // 统计每列空值数量
            System.out.println(column + ": " + nullCount);
        }

        // 4. 数据清洗
        Dataset<Row> cleanedDF = steamDF
                .dropDuplicates()  // 删除重复行
                .na().drop()       // 删除包含空值的行
                // 类型转换（确保后续计算使用正确类型）
                .withColumn("positive_ratings", col("positive_ratings").cast("int"))
                .withColumn("negative_ratings", col("negative_ratings").cast("int"))
                .withColumn("average_playtime", col("average_playtime").cast("double"))
                .withColumn("median_playtime", col("median_playtime").cast("double"))
                .withColumn("price", col("price").cast("double"))
                .filter(col("price").isNotNull().and(col("price").gt(0))); // 过滤无效价格

        // 5. 特征工程
        Dataset<Row> transformedDF = cleanedDF
                // 创建新特征
                .withColumn("total_ratings", col("positive_ratings").plus(col("negative_ratings"))) // 总评价数
                .withColumn("rating_ratio", col("positive_ratings").divide(col("total_ratings")))   // 好评率
                // 基于好评率创建分类标签
                .withColumn("rating_category",
                        when(col("rating_ratio").geq(0.95), lit("好评如潮"))
                                .when(col("rating_ratio").geq(0.8).and(col("rating_ratio").lt(0.95)), lit("特别好评"))
                                .when(col("rating_ratio").geq(0.7).and(col("rating_ratio").lt(0.8)), lit("多半好评"))
                                .when(col("rating_ratio").geq(0.4).and(col("rating_ratio").lt(0.7)), lit("褒贬不一"))
                                .when(col("rating_ratio").geq(0.2).and(col("rating_ratio").lt(0.4)), lit("多半差评"))
                                .otherwise(lit("差评如潮")))
                // 创建二分类标签（用于机器学习）
                .withColumn("label",
                        when(col("negative_ratings").equalTo(0), 0)  // 无差评=0
                                .when(col("positive_ratings").divide(col("negative_ratings")).gt(10), 1) // 好评差评比>10=1
                                .otherwise(0));                      // 其他情况=0

        // 6. 类别特征编码
        // 6.1 将平台字符串转换为数字索引
        StringIndexer indexer = new StringIndexer()
                .setInputCol("platforms")        // 输入列
                .setOutputCol("platformsIndex"); // 输出索引列
        Dataset<Row> indexedDF = indexer.fit(transformedDF).transform(transformedDF);

        // 6.2 对索引进行独热编码
        OneHotEncoder encoder = new OneHotEncoder()
                .setInputCols(new String[]{"platformsIndex"})
                .setOutputCols(new String[]{"platformsVec"}); // 输出向量列
        OneHotEncoderModel encoderModel = encoder.fit(indexedDF);
        Dataset<Row> encodedDF = encoderModel.transform(indexedDF);

        // 7. 数值特征处理
        // 7.1 将平均游戏时长转换为向量（供标准化使用）
        VectorAssembler playtimeVecAssembler = new VectorAssembler()
                .setInputCols(new String[]{"average_playtime"})
                .setOutputCol("average_playtime_vec");
        Dataset<Row> vecDF = playtimeVecAssembler.transform(encodedDF);

        // 7.2 标准化游戏时长特征（均值为0，标准差为1）
        StandardScaler scaler = new StandardScaler()
                .setInputCol("average_playtime_vec")
                .setOutputCol("scaled_playtime")      // 标准化后输出列
                .setWithStd(true);                    // 启用标准差缩放
        Dataset<Row> scaledDF = scaler.fit(vecDF).transform(vecDF);

        // 8. 合并所有特征向量
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                    "scaled_playtime",    // 标准化后的游戏时长
                    "price",              // 价格
                    "total_ratings",      // 总评价数
                    "platformsVec"        // 编码后的平台向量
                })
                .setOutputCol("rawFeatures") // 最终特征向量
                .setHandleInvalid("keep");   // 保留无效值
        Dataset<Row> assembledDF = assembler.transform(scaledDF);

        // 9. 释放资源
        spark.stop();
    }
}
