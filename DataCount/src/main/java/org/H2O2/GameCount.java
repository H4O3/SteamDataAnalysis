package org.H2O2;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

import java.util.Properties;

public class GameCount {
    /**
     * 主函数：租金数据分析及存储程序入口
     * 功能说明：
     * 1. 创建Spark会话进行数据处理
     * 2. 对原始租房数据进行清洗和转换
     * 3. 分析不同行政区的平均租金
     */
    public static void main(String[] args) {
        // 初始化Spark环境配置（本地模式，使用所有可用核心）
        SparkSession spark = SparkSession.builder().master("local[*]").appName("RentTest").getOrCreate();
        Dataset<Row> df = spark.read().option("header", true).option("inferSchema", true).csv("steam.csv");

        // 数据预处理：去除完全重复的数据记录

        // 配置MySQL数据库连接参数
        String url = "jdbc:mysql://localhost:3306/atguigudb";
        Properties properties = new Properties();
        properties.put("user", "root");
        properties.put("password", "123456");
        properties.put("driver", "com.mysql.cj.jdbc.Driver");

        //1. 分析最受欢迎的游戏类型（Genres）
        //你可以统计每种游戏类型的出现频率，从而判断哪些类型更受欢迎。
        // 对 "genres" 字段进行拆分并统计每个类型的数量
        Dataset<Row> genreCounts = df.selectExpr("explode(split(genres, ';')) as genre").groupBy("genre").count().orderBy(functions.desc("count"));
        // 展示结果
        genreCounts.show();

        //2. 分析不同平台（Windows/Mac/Linux）上的游戏分布
        //你可以在 platforms 字段中解析出支持的平台，并统计各平台上的游戏数量。
        // 拆分 "platforms" 字段并统计每个平台的数量
        Dataset<Row> platformCounts = df.selectExpr("explode(split(platforms, ';')) as platform").groupBy("platform").count().orderBy(functions.desc("count"));
        // 展示结果
        platformCounts.show();

        //3. 分析正面评价与负面评价的比例
        //你可以通过 positive_ratings 和 negative_ratings 字段计算每款游戏的正负评价比例。
        // 计算每款游戏的正负评价比例
        Dataset<Row> ratingRatio = df.withColumn("ratio", df.col("positive_ratings").cast("double").divide(df.col("negative_ratings").cast("double")));
        // 展示结果
        ratingRatio.select("name", "positive_ratings", "negative_ratings", "ratio").show();

        //4. 分析价格区间分布
        //你可以将游戏价格划分为多个区间（例如：0-10、10-50、50+），然后统计每个区间的数量。
        // 使用 when/otherwise 表达式定义价格区间
        Dataset<Row> priceDistribution = df.withColumn("price_range", functions.when(df.col("price").leq(10), "0-10").when(df.col("price").gt(10).and(df.col("price").leq(50)), "10-50").otherwise("50+"));
        // 统计每个价格区间的数量
        Dataset<Row> priceRangeCount = priceDistribution.groupBy("price_range").count();
        // 展示结果
        priceRangeCount.show();

        //5. 分析开发商（Developer）的游戏数量
        //你可以统计每个开发商开发的游戏数量，以了解哪些开发商产出较多
        // 按照 "developer" 分组并统计数量
        Dataset<Row> developerCounts = df.groupBy("developer").count().orderBy(functions.desc("count"));
        // 展示结果
        developerCounts.show();

        //6. 分析游戏拥有者数量（Owners）与平均游玩时间（Average Playtime）的关系
        //你可以分析拥有者数量与平均游玩时间之间的关系，尝试找出两者是否存在关联。
        // 选择相关字段并展示前几行数据
        Dataset<Row> ownerPlaytimeRelation = df.select("name", "owners", "average_playtime");
        // 展示结果
        ownerPlaytimeRelation.show();

        //7. 计算每款游戏的“评分效率”
        //定义：正向评价数与负面评价数的比值（若负面评价为 0，则设为 positive_ratings 的最大值）。
        // 防止除零错误，计算评分效率
        Dataset<Row> ratingEfficiency = df.withColumn("efficiency",
                functions.when(df.col("negative_ratings").equalTo(0), 100000)
                        .otherwise(df.col("positive_ratings").divide(df.col("negative_ratings"))));
        // 展示结果
        ratingEfficiency.select("name", "positive_ratings", "negative_ratings", "efficiency").show();

        //8. 分析不同发行年份的游戏数量
        //你可以通过解析 release_date 字段提取年份，并统计每年发布的游戏数量。
        // 提取年份并分组统计
        Dataset<Row> releaseYearCount = df.withColumn("year", functions.substring(df.col("release_date"), 0, 4))
                .groupBy("year")
                .count()
                .orderBy(functions.desc("count"));
        // 展示结果
        releaseYearCount.show();

        //9. 分析拥有者数量最多的前 10 款游戏
        //你可以使用 owners 字段进行排序并选出前 10 款游戏。
        // 按照 owners 排序并取前 10
        Dataset<Row> topOwners = df.select("name", "owners")
                .orderBy(functions.desc("owners"))
                .limit(10);
        // 展示结果
        topOwners.show();

        //10. 分析每个开发商的平均游戏价格
        //你可以按照 developer 分组，并计算其开发游戏的平均价格。
        // 计算每个开发商的平均价格
        Dataset<Row> avgPricePerDeveloper = df.groupBy("developer")
                .agg(functions.avg("price").alias("avg_price"))
                .orderBy(functions.desc("avg_price"));
        // 展示结果
        avgPricePerDeveloper.show();

        //11. 分析哪些游戏支持中文
        //你可以检查 english 字段是否为 1 来判断是否支持英文，同理也可以扩展到其他语言字段。
        // 过滤出支持英文的游戏
        Dataset<Row> englishSupportedGames = df.filter(df.col("english").equalTo(1))
                .select("name");
        // 展示结果
        englishSupportedGames.show();

        //12. 分析每个游戏类别的平均价格
        //你可以对 categories 字段进行拆分后，统计每个类别下的平均价格。
        // 拆分 categories 并计算每个类别的平均价格
        Dataset<Row> categoryAvgPrice = df.selectExpr("explode(split(categories, ';')) as category", "price")
                .groupBy("category")
                .agg(functions.avg("price").alias("avg_price"))
                .orderBy(functions.desc("avg_price"));
        // 展示结果
        categoryAvgPrice.show();

        //13. 分析游戏成就数量与用户评价的关系
        //你可以通过 achievements 字段分析游戏成就数量与用户评价之间的关系。
        // 选择相关字段并展示数据
        Dataset<Row> achievementRatingRelation = df.select("name", "achievements", "positive_ratings", "negative_ratings");
        // 展示结果
        achievementRatingRelation.show();


    }
}
