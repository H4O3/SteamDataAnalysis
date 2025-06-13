package org.H2O2;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

// 静态导入 Spark SQL 函数以便在 DataFrame 操作中使用
import static org.apache.spark.sql.functions.*;

public class AudiencePreferenceModel {
    public static void main(String[] args) {

        // 初始化 Spark 会话
        // 创建 SparkSession 实例并设置应用程序名称、运行模式等
        SparkSession spark = SparkSession.builder()
                .appName("Audience Preference Model") // 设置应用名称
                .master("local[*]") // 在本地运行，并利用所有核心
                .getOrCreate(); // 创建或获取现有的 SparkSession

        // 读取 CSV 数据集
        // 使用 SparkSession 读取 steam.csv 文件，自动推断 schema 和包含 header
        Dataset<Row> df = spark.read()
                .option("header", true) // CSV 文件有表头
                .option("inferSchema", true) // 自动推断数据类型
                .csv("steam.csv"); // 数据源路径

        // 删除包含任何空值的行
        df = df.na().drop(); // 移除存在 null 值的行

        // 创建目标变量 label：
        // 如果 negative_ratings 为 0，则 label 为 0；
        // 如果 positive_ratings / negative_ratings > 10，则 label 为 1；
        // 其他情况 label 为 0。
        // 1表示受欢迎
        Dataset<Row> labeledDf = df.withColumn(
                "label",
                when(col("negative_ratings").equalTo(0), 0)
                        .when(col("positive_ratings").divide(col("negative_ratings")).gt(10), 1)
                        .otherwise(0)
        );

        // 拆分 genres 字段（按 ';' 分隔）生成数组列 genreArray
        RegexTokenizer tokenizer = new RegexTokenizer()
                .setInputCol("genres") // 输入列为 genres
                .setOutputCol("genreArray") // 输出列为 genreArray
                .setPattern(";"); // 分割符为 ';'

        Dataset<Row> tokenizedDf = tokenizer.transform(labeledDf); // 对数据集执行转换操作

        // 展开 genreArray 数组为多行（每个 genre 占一行）
        Dataset<Row> explodedDf = tokenizedDf.withColumn("genre", explode(col("genreArray")));

        // 对单个 genre 进行 StringIndexer 操作，将字符串类别转换为数值索引
        StringIndexer genreIndexer = new StringIndexer()
                .setInputCol("genre") // 输入列为 genre
                .setOutputCol("indexedGenre"); // 输出列为 indexedGenre

        Dataset<Row> indexedGenresDf = genreIndexer.fit(explodedDf).transform(explodedDf); // 训练并应用模型

        // 使用 OneHotEncoder 编码处理索引化的 genre 列
        OneHotEncoder encoder = new OneHotEncoder()
                .setInputCol("indexedGenre") // 输入列为 indexedGenre
                .setOutputCol("genreVec"); // 输出列为 genreVec

        OneHotEncoderModel encoderModel = encoder.fit(indexedGenresDf); // 训练编码器
        Dataset<Row> encodedDf = encoderModel.transform(indexedGenresDf); // 应用编码器到数据集

        // 将 price 和 achievements 列转换为 double 类型以确保后续计算正确
        Dataset<Row> numericDf = encodedDf
                .withColumn("price", col("price").cast("double")) // 转换 price 为 double
                .withColumn("achievements", col("achievements").cast("double")); // achievements 同理

        // 合并特征向量：将多个特征列合并为一个 rawFeatures 列
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"genreVec", "price", "achievements"}) // 特征输入列
                .setOutputCol("rawFeatures") // 输出列名
                .setHandleInvalid("skip"); // 忽略无效值

        Dataset<Row> assembledDf = assembler.transform(numericDf); // 执行特征合并

        // 标准化 rawFeatures 列，使不同量纲的特征具有可比性
        StandardScaler scaler = new StandardScaler()
                .setInputCol("rawFeatures") // 输入列为 rawFeatures
                .setOutputCol("features"); // 输出标准化后的 features

        Dataset<Row> scaledDf = scaler.fit(assembledDf).transform(assembledDf); // 标准化数据
        scaledDf.show();

        // 拆分数据集为训练集 (80%) 和测试集 (20%)
        Dataset<Row>[] splits = scaledDf.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> train = splits[0]; // 训练数据集
        Dataset<Row> test = splits[1]; // 测试数据集

        // 训练逻辑回归模型
        LogisticRegression lr = new LogisticRegression()
                .setMaxIter(10) // 最大迭代次数
                .setRegParam(0.3) // 正则化参数
                .setElasticNetParam(0.8); // ElasticNet 混合参数

        LogisticRegressionModel model = lr.fit(train); // 拟合训练数据

        // 使用训练好的模型对测试集进行预测
        Dataset<Row> predictions = model.transform(test);
        predictions.select("prediction", "label", "features").show(5); // 显示前 5 条结果

        // 评估模型准确性
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label") // 真实标签列
                .setPredictionCol("prediction") // 预测结果列
                .setMetricName("accuracy"); // 评估指标为准确率
        double accuracy = evaluator.evaluate(predictions); // 计算准确率
        System.out.println("Test Accuracy: " + accuracy); // 输出准确率

        // 关闭 Spark 会话，释放资源
        spark.stop();
    }
}

