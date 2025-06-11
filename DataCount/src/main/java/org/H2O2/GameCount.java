package org.H2O2;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Properties;

public class GameCount {
    /**
     * 主函数：租金数据分析及存储程序入口
     * 功能说明：
     * 1. 创建Spark会话进行数据处理
     * 2. 对原始租房数据进行清洗和转换
     * 3. 分析不同行政区的平均租金
     * 4. 将结果持久化到MySQL数据库
     */
    public static void main(String[] args) {
        // 初始化Spark环境配置（本地模式，使用所有可用核心）
        SparkSession spark = SparkSession.builder().master("local[*]").appName("RentTest").getOrCreate();


        Dataset<Row> df = spark.read().option("header", true).option("inferSchema", true).csv("steam.csv");

        // 数据预处理：去除完全重复的数据记录
        Dataset<Row> rowDataset = df.dropDuplicates();

        // 配置MySQL数据库连接参数
        String url = "jdbc:mysql://localhost:3306/atguigudb";
        Properties properties = new Properties();
        properties.put("user", "root");
        properties.put("password", "123456");
        properties.put("driver", "com.mysql.cj.jdbc.Driver");


    }
}
