### Steam游戏数据列整理

#### 一、原始数据列（来自steamTest.csv）
| 列名 | 含义 |
|------|------|
| appid | 游戏在Steam平台上的唯一标识ID |
| name | 游戏名称 |
| release_date | 游戏发布日期（格式YYYY-MM-DD） |
| english | 是否支持英文（1=支持，0=不支持） |
| developer | 游戏开发商名称 |
| publisher | 游戏发行商名称 |
| platforms | 支持的游戏平台（多个平台用分号分隔，如windows;mac） |
| required_age | 游戏要求的最低玩家年龄 |
| categories | 游戏所属分类（多个分类用分号分隔） |
| genres | 游戏所属流派（多个流派用分号分隔） |
| steamspy_tags | SteamSpy平台标记的游戏标签 |
| achievements | 游戏内成就数量 |
| positive_ratings | 游戏好评数量 |
| negative_ratings | 游戏差评数量 |
| average_playtime | 玩家平均游戏时长（分钟） |
| median_playtime | 玩家游戏时长中位数（分钟） |
| owners | 游戏拥有者数量范围（格式下限-上限，如5000000-10000000） |
| price | 游戏价格（美元） |

#### 二、处理过程中新增的特征列（在DataProcessing.java中生成）
| 列名 | 含义 |
|------|------|
| release_year | 从release_date提取的发布年份 |
| total_ratings | 总评价数（计算公式：positive_ratings + negative_ratings） |
| rating_ratio | 好评率（计算公式：positive_ratings / total_ratings） |
| rating_category | 基于好评率划分的类别（如"好评如潮"、"多半差评"等） |
| label | 二分类标签（0=评价较差，1=评价极好） |
| platformsIndex | 平台字符串转换成的数字索引 |
| platformsVec | 平台索引的独热编码向量 |
| average_playtime_vec | 平均游戏时长转换成的向量（供标准化使用） |
| scaled_playtime | 标准化后的平均游戏时长（均值为0，标准差为1） |
| rawFeatures | 合并所有特征的最终向量（包含价格、总评价数、平台向量等） |

#### 三、关键处理说明
1. **数据清洗规则**
    - required_age空值填充为0
    - developer/publisher空值填充为"Unknown"
    - average_playtime超过100,000或负值时会被修正

2. **特征工程**
    - 类别特征通过StringIndexer和OneHotEncoder转换为数值向量
    - 数值特征通过StandardScaler标准化处理

3. **输出规则**
    - 最终输出文件包含原始列+新增特征列
    - 所有列均转为字符串类型保存