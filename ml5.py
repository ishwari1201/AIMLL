
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

# Step 1: Take dataset filename from user
filename = input("Enter CSV filename: ").strip().strip('"')

# Step 2: Read dataset
data = pd.read_csv(filename, header=None)
data = data.fillna("")

# Step 3: Convert dataset into transactions
transactions = []

for i in range(len(data)):
    transaction = []
    for item in data.iloc[i]:
        if str(item).strip() != "":
            transaction.append(str(item).strip().lower())
    transactions.append(transaction)

print("\nTotal transactions:", len(transactions))

# Step 4: Take Apriori parameters from user
support = float(input("Enter minimum support: "))
confidence = float(input("Enter minimum confidence: "))
lift = float(input("Enter minimum lift: "))

# Step 5: Function to generate frequent itemsets and rules
def generate_rules():
    results = list(apriori(
        transactions,
        min_support=support,
        min_confidence=confidence,
        min_lift=lift,
        min_length=2
    ))

    rules = []

    for result in results:
        print("\nFrequent Itemset:", list(result.items))
        print("Support:", result.support)

        for stat in result.ordered_statistics:
            if stat.items_base and stat.items_add:
                rules.append({
                    "If": ", ".join(stat.items_base),
                    "Then": ", ".join(stat.items_add),
                    "Support": result.support,
                    "Confidence": stat.confidence,
                    "Lift": stat.lift
                })

    return pd.DataFrame(rules)

# Step 6: Generate association rules
rules_df = generate_rules()

# Step 7: Display and evaluate rules
if rules_df.empty:
    print("\nNo rules found. Try lowering the minimum support or confidence.")

else:
    rules_df = rules_df.sort_values(
        by="Confidence", ascending=False
    )

    print("\nAssociation Rules:")
    print(rules_df.to_string(index=False))

    print("\nEvaluation Metrics:")
    print("Total Association Rules:", len(rules_df))
    print("Average Support:", rules_df["Support"].mean())
    print("Average Confidence:", rules_df["Confidence"].mean())
    print("Average Lift:", rules_df["Lift"].mean())

    # Step 8: Save rules to CSV
    rules_df.to_csv("association_rules.csv", index=False)
    print("\nRules saved to association_rules.csv")

    # Step 9: Plot one visualization
    plt.figure(figsize=(8, 5))
    plt.scatter(
        rules_df["Support"],
        rules_df["Confidence"]
    )
    plt.xlabel("Support")
    plt.ylabel("Confidence")
    plt.title("Support vs Confidence")
    plt.show()

# Step 10: Conclusion
print("\nApriori Algorithm Applied Successfully!")
