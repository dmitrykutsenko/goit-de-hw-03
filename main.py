from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, round
import matplotlib.pyplot as plt
import pandas as pd

# Створення Spark Session
spark = SparkSession.builder \
    .appName("GoIT_HW_03") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

def save_df_as_image(spark_df, filename, rows=20):
    pdf = spark_df.limit(rows).toPandas()

    fig, ax = plt.subplots(
        figsize=(max(8, len(pdf.columns) * 2), max(2, rows * 0.5))
    )

    ax.axis("off")

    table = ax.table(
        cellText=pdf.values,
        colLabels=pdf.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------
# 1. Завантаження CSV-файлів
# ---------------------------

users_df = spark.read.csv(
    "data/users.csv",
    header=True,
    inferSchema=True
)

purchases_df = spark.read.csv(
    "data/purchases.csv",
    header=True,
    inferSchema=True
)

products_df = spark.read.csv(
    "data/products.csv",
    header=True,
    inferSchema=True
)


# ---------------------------
# 2. Очищення даних
# ---------------------------

users_df = users_df.dropna()
purchases_df = purchases_df.dropna()
products_df = products_df.dropna()


# ---------------------------
# Join таблиць
# ---------------------------

df = purchases_df \
    .join(users_df, "user_id") \
    .join(products_df, "product_id")

# Вартість покупки
df = df.withColumn(
    "purchase_amount",
    round(col("quantity") * col("price"), 2)
)

print("\n=== Data після JOIN ===")
df.show(10, truncate=False)

save_df_as_image(
    df.select(
        "user_id",
        "product_id",
        "category",
        "quantity",
        "price",
        "purchase_amount"
    ),
    "00_join_result.png",
    rows=10
)


# ---------------------------
# 3. Загальна сума покупок по категоріях
# ---------------------------

total_by_category = df.groupBy("category") \
    .agg(
        round(
            sum("purchase_amount"),
            2
        ).alias("total_amount")
    )

print("\n=== Загальна сума покупок по категоріях ===")
total_by_category.show(truncate=False)

save_df_as_image(
    total_by_category,
    "01_total_by_category.png"
)

# ---------------------------
# 4. Витрати вікової групи 18-25 років
# ---------------------------

young_df = df.filter(
    (col("age") >= 18) &
    (col("age") <= 25)
)

young_by_category = young_df.groupBy("category") \
    .agg(
        round(
            sum("purchase_amount"),
            2
        ).alias("age_18_25_total")
    )

print("\n=== Сума покупок 18-25 років ===")
young_by_category.show(truncate=False)

save_df_as_image(
    young_by_category,
    "02_age_18_25_total.png"
)


# ---------------------------
# 5. Частка витрат
# ---------------------------

young_total = young_df.agg(
    sum("purchase_amount").alias("total")
).collect()[0]["total"]

share_df = young_by_category.withColumn(
    "percentage",
    round(
        col("age_18_25_total") / young_total * 100,
        2
    )
)

print("\n=== Відсоток витрат по категоріях ===")
share_df.show(truncate=False)

save_df_as_image(
    share_df,
    "03_category_percentage.png"
)


# ---------------------------
# 6. Top-3 категорії
# ---------------------------

top3 = share_df.orderBy(
    col("percentage").desc()
).limit(3)

print("\n=== TOP-3 категорії ===")
top3.show(truncate=False)

save_df_as_image(
    top3,
    "04_top3_categories.png"
)

spark.stop()