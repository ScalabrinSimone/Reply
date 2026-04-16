from agent import run_fraud_detection

if __name__ == "__main__":
    fraud_ids = run_fraud_detection()
    print("\n=== TRANSAZIONI FRAUDOLENTE ===")
    for tid in fraud_ids:
        print(tid)
