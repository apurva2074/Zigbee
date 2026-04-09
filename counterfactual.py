import pandas as pd

def generate_counterfactuals(sample, model):
    suggestions = []

    base_pred = model.predict(pd.DataFrame([sample]))[0]

    # Try reducing lights
    new_sample = sample.copy()
    new_sample['lights'] = max(0, sample['lights'] - 10)

    new_pred = model.predict(pd.DataFrame([new_sample]))[0]

    suggestions.append({
        "change": f"Reduce lights to {new_sample['lights']}",
        "energy": round(new_pred, 2),
        "saving": round(base_pred - new_pred, 2)
    })

    # Try lowering temperature
    new_sample = sample.copy()
    new_sample['T1'] = sample['T1'] - 1

    new_pred = model.predict(pd.DataFrame([new_sample]))[0]

    suggestions.append({
        "change": f"Reduce temperature to {new_sample['T1']}",
        "energy": round(new_pred, 2),
        "saving": round(base_pred - new_pred, 2)
    })

    return suggestions