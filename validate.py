import pandas as pd
import ast

sub  = pd.read_csv(r"c:\Users\rithu\OneDrive\Desktop\hcl\submission.csv")
test = pd.read_csv(r"c:\Users\rithu\OneDrive\Desktop\hcl\c215051c-6-Archive 4\test.csv")

print("=== SUBMISSION VALIDATION ===")
print("Shape:", sub.shape, " (expected 10977 x 2)")
print("Columns:", list(sub.columns))
print("Index range:", sub["Index"].min(), "to", sub["Index"].max())
print("Matches test index:", (sub["Index"].values == test["Index"].values).all())

lengths = sub["Index_list"].apply(lambda x: len(ast.literal_eval(x)))
print("All lists have 10 items:", (lengths == 10).all())
print("List length stats — min:", lengths.min(), "max:", lengths.max())

print("\nFirst 5 rows:")
print(sub.head(5).to_string())
print("\nLast 3 rows:")
print(sub.tail(3).to_string())
