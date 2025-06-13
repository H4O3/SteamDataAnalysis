在代码中，游戏的**受欢迎程度（`label`）** 主要与以下字段有关：

### 1. **正面评价数（`positive_ratings`）**
- 表示玩家对游戏的好评数量。
- 在代码中，通过 `positive_ratings / negative_ratings` 的比值来判断游戏是否受欢迎。
- 比值越大，说明游戏越受好评。

### 2. **负面评价数（`negative_ratings`）**
- 表示玩家对游戏的差评数量。
- 在代码中，`positive_ratings` 和 `negative_ratings` 的比值被用来定义游戏是否“流行”：
  ```java
  when(col("negative_ratings").equalTo(0), 0)
      .when(col("positive_ratings").divide(col("negative_ratings")).gt(10), 1)
      .otherwise(0)
  ```

    - 如果比值大于 10，则标记为 `1`（表示热门游戏）；
    - 否则标记为 `0`（非热门游戏）。

### 3. **游戏类型（`genres`）**
- 游戏的类型信息通过以下步骤影响模型：
    1. 使用 `RegexTokenizer` 将 `genres` 字段按分号 `;` 分割成数组列 `genreArray`。
    2. 使用 `explode` 展开 `genreArray`，将每种类型单独拆分成行。
    3. 使用 `StringIndexer` 对每个游戏类型进行编码（如动作、冒险等），生成 `indexedGenre`。
    4. 再使用 `OneHotEncoder` 编码 `indexedGenre`，生成 `genreVec`，将其作为特征输入模型。
- 这些操作表明不同游戏类型可能对游戏受欢迎程度产生影响。

### 4. **价格（`price`）**
- 游戏的价格被转换为 `double` 类型，并与其他特征合并后输入模型。
- 价格可能会直接影响玩家购买意愿，从而影响游戏的受欢迎程度。

### 5. **成就数量（`achievements`）**
- 成就数量也被转换为 `double` 类型，并作为特征输入模型。
- 游戏中的成就数量反映了游戏的可玩性和深度，这可能间接影响玩家的评价和游戏的受欢迎程度。

### 6. **特征向量（`features`）**
- 所有上述特征（`genreVec`、`price` 和 `achievements`）被合并并标准化为一个特征向量 `features`。
- 这个特征向量最终被用于训练逻辑回归模型，以预测游戏的受欢迎程度。

### 总结
游戏的受欢迎程度主要与以下几个因素相关：
- 玩家对游戏的评价（正面和负面评价数量）。
- 游戏的类型（例如动作、冒险等）。
- 游戏的价格。
- 游戏中的成就数量。

这些因素通过特征工程处理后，作为输入提供给机器学习模型，以预测游戏是否受欢迎。


**appid**
整数
Steam 应用程序的唯一 ID，用于标识每款游戏或软件。

**name**
字符串
游戏名称。

**release_date**
日期
游戏的发行日期（格式为 YYYY-MM-DD）。

**english**
布尔值
表示该游戏是否支持英文（1 = 支持，0 = 不支持）。

**developer**
字符串
游戏开发公司或团队名称。

**publisher**
字符串
游戏发行商名称。

**platforms**
字符串
游戏支持的平台（例如 Windows、Mac、Linux），多个平台以分号分隔。

**required_age**
整数
玩家需要达到的最小年龄才能游玩该游戏（0 表示无年龄限制）。

**categories**
字符串
游戏分类标签，描述游戏功能（如单人、多人、在线对战等），以分号分隔。

**genres**
字符串
游戏所属的类型（如动作、冒险、角色扮演等），以分号分隔。

**steamspy_tags**
字符串
根据玩家行为推测的游戏标签，反映玩家社区认为该游戏的主要特点。

**achievements**
整数
游戏中的成就数量。

**positive_ratings**
整数
用户对该游戏的正面评价数量。

**negative_ratings**
整数
用户对该游戏的负面评价数量。

**average_playtime**
整数
所有玩家的平均游戏时长（单位：分钟）。

**median_playtime**
整数
玩家中间值的游戏时长（单位：分钟）。

**owners**
字符串
拥有该游戏的用户数量范围（例如 10000000-20000000）。

**price**
浮点数
游戏的价格（单位：美元）。

**label**
整数
目标变量，根据 positive_ratings / negative_ratings 的比值计算得出：<br> - 如果比值 > 10，则标记为 1（表示热门游戏）<br> - 否则标记为 0（非热门游戏）

**genreArray**
数组
将原始 genres 字段按分号 ; 分割后的数组形式。

**genre**
字符串
将 genreArray 展开后得到的单个游戏类型。

**indexedGenre**
浮点数
使用 StringIndexer 对 genre 进行编码后得到的数值索引。

**genreVec**
向量
使用 One-Hot 编码将 indexedGenre 转换为二进制向量。

**rawFeatures**
向量
合并所有特征（包括 genreVec、price 和 achievements）后得到的原始特征向量。

**features**
向量
经过标准化处理后的特征向量，用于模型训练和预测。
