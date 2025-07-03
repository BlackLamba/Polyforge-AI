import pandas as pd

# Загрузка
df_diff = pd.read_csv("data/Solvent_Diffusivity_NCOMMS/ML_Diffusivity.csv") # с logD
df_sorp = pd.read_csv("data/Solvent_Sorption_NCOMMS/ML_uptake_sorption.csv") # с log10(mmol_solvent/g_Polymer)

# Объединение по SMILES полимера и растворителя
merged = pd.merge(
    df_diff,
    df_sorp[["Polymer_SMILES", "Solvent_SMILES", "log10(mmol_solvent/g_Polymer)"]],
    on=["Polymer_SMILES", "Solvent_SMILES"],
    how="outer"
)

# Сохранение
merged.to_csv("polyverse_merged_dataset.csv", index=False)
print(f"Объединено строк: {len(merged)}")
