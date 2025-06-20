package org.H2O2;

import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;

import java.util.Arrays;

import static org.apache.spark.sql.functions.*;

public class DataProcessing {
    public static void main(String[] args) {
        // 1. 初始化Spark会话
        SparkSession spark = SparkSession.builder()
                .appName("Audience Preference Model") // 应用名称
                .master("local[*]")                  // 本地模式使用所有CPU核心
                .getOrCreate();

        // 2. 读取本地CSV数据源
        Dataset<Row> steamDF = spark.read()
                .option("header", true)     // 首行为列名
                .option("inferSchema", true) // 自动推断数据类型
                .csv("steam.csv");          // 数据文件路径

//        //2. 读取HDFS数据源
//        Dataset<Row> steamDF = spark.read()
//                .option("header", true)
//                .option("inferSchema", true)
//                .csv("hdfs://192.168.88.161:8020/test/input/steam.csv"); // HDFS路径

        // 3. 数据探索
        steamDF.describe().show();  // 显示数值型列的统计摘要
        System.out.println("缺失值统计:");
        for (String column : steamDF.columns()) {
            long nullCount = steamDF.filter(col(column).isNull()).count(); // 统计每列空值数量
            System.out.println(column + ": " + nullCount);
        }

        // 4. 数据清洗
        Dataset<Row> cleanedDF = steamDF
                // 4.1 删除重复行（根据关键字段）
                .dropDuplicates(new String[]{"appid", "name"})  // 根据游戏ID和名称去重

                // 4.2 处理空值
                .na().drop(new String[]{"appid", "name"})       // 关键字段空值直接删除
                .na().fill(0, new String[]{"required_age"})     // 年龄要求空值填充0
                .na().fill("Unknown", new String[]{"developer", "publisher"}) // 开发商/发行商空值填充

                // 4.3 类型转换与验证
                .withColumn("positive_ratings",
                        when(col("positive_ratings").cast("int").isNotNull(), col("positive_ratings").cast("int"))
                                .otherwise(0))  // 非法值设为0
                .withColumn("negative_ratings",
                        when(col("negative_ratings").cast("int").isNotNull(), col("negative_ratings").cast("int"))
                                .otherwise(0))
                .withColumn("price",
                        when(col("price").cast("double").geq(0), col("price").cast("double")) // 价格非负
                                .otherwise(0.0))

                // 4.4 异常值处理
                .withColumn("average_playtime",
                        when(col("average_playtime").gt(100000), 100000) // 设置游戏时长上限
                                .when(col("average_playtime").lt(0), 0)          // 负值修正为0
                                .otherwise(col("average_playtime").cast("double")))

                // 4.5 数据规范化
                .withColumn("release_year", year(to_date(col("release_date"), "yyyy-MM-dd"))) // 提取发布年份
                .filter(col("release_year").between(1990, 2023))  // 过滤无效年份

                // 4.6 平台字段拆分
                .withColumn("platforms",
                        when(col("platforms").isNull(), "windows")    // 空值默认windows
                                .otherwise(col("platforms")));

        // 4.7 异常值统计
        System.out.println("异常值处理统计:");
        long negativePlaytime = cleanedDF.filter(col("average_playtime").lt(0)).count();
        long excessivePlaytime = cleanedDF.filter(col("average_playtime").gt(100000)).count();
        System.out.println("负游戏时长记录: " + negativePlaytime);
        System.out.println("超长游戏时长记录: " + excessivePlaytime);

        // 4.8 价格分布分析
        cleanedDF.select(mean("price").alias("avg_price"),
                min("price").alias("min_price"),
                max("price").alias("max_price")).show();

        //效果检查
        System.out.println("缺失值统计:");
        for (String column : steamDF.columns()) {
            long nullCount = steamDF.filter(col(column).isNull()).count(); // 统计每列空值数量
            System.out.println(column + ": " + nullCount);
        }

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
                .setOutputCol("average_playtime_vec")
                .setHandleInvalid("skip");
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


        // 9. 保存处理后的数据为CSV
        Column[] stringColumns = Arrays.stream(assembledDF.columns())
                .map(colName -> col(colName).cast(DataTypes.StringType).as(colName))
                .toArray(Column[]::new);

        Dataset<Row> stringDF = assembledDF.select(stringColumns);

        stringDF.coalesce(1).write()
                .option("header", true)
                .mode("overwrite")
                .csv("output2/processed_steam_data_string");

        // 释放资源
        spark.stop();
    }
}
