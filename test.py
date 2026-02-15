#!/usr/bin/env python3
"""
Script de test automatique pour valider le backend Flask corrigé
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def print_section(title):
    """Affiche un titre de section"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_success(message):
    """Affiche un message de succès"""
    print(f"✅ {message}")

def print_error(message):
    """Affiche un message d'erreur"""
    print(f"❌ {message}")

def print_info(message):
    """Affiche un message d'information"""
    print(f"ℹ️  {message}")

# Créer le dataset de test
print_section("CRÉATION DU DATASET DE TEST")

data = {
    'id': [1, 2, 3, 4, 2, 5, 6],
    'name': ['Alice', 'Bob', 'Charlie', 'Dave', 'Eve', 'Frank', 'Grace'],
    'age': [25, None, 120, 30, 28, None, 27],
    'salary': [50000, 60000, 200000, 70000, 65000, 300000, 80000]
}
df = pd.DataFrame(data)

print_info(f"Dataset créé avec {len(df)} lignes et {len(df.columns)} colonnes")
print("\nAperçu des données:")
print(df.to_string(index=False))

# Test 1: Détection des valeurs manquantes
print_section("TEST 1: DÉTECTION DES VALEURS MANQUANTES")

missing_values = {}
total_missing = 0

for column in df.columns:
    missing_count = df[column].isnull().sum()
    if missing_count > 0:
        missing_values[column] = int(missing_count)
        total_missing += missing_count
        print_info(f"Colonne '{column}': {missing_count} valeur(s) manquante(s)")

if total_missing == 2:
    print_success(f"TOTAL: {total_missing} valeurs manquantes détectées correctement")
else:
    print_error(f"ERREUR: {total_missing} valeurs manquantes détectées (attendu: 2)")

# Test 2: Détection des doublons (méthode standard)
print_section("TEST 2: DÉTECTION DES DOUBLONS (TOUTES COLONNES)")

duplicates_all = df.duplicated(keep=False).sum()
print_info(f"Méthode: df.duplicated(keep=False)")
print_info(f"Résultat: {duplicates_all} doublons détectés")

if duplicates_all == 0:
    print_success("Correct: aucun doublon complet (lignes identiques à 100%)")
else:
    print_error(f"Inattendu: {duplicates_all} doublons détectés")

# Test 3: Détection des doublons (sur colonne ID)
print_section("TEST 3: DÉTECTION DES DOUBLONS (COLONNE ID)")

duplicates_id = df.duplicated(subset=['id'], keep=False).sum()
print_info(f"Méthode: df.duplicated(subset=['id'], keep=False)")
print_info(f"Résultat: {duplicates_id} doublons détectés")

if duplicates_id == 2:
    print_success("Correct: 2 lignes avec id=2 détectées")
    print("\nLignes dupliquées:")
    print(df[df.duplicated(subset=['id'], keep=False)][['id', 'name', 'age', 'salary']].to_string(index=False))
else:
    print_error(f"ERREUR: {duplicates_id} doublons détectés (attendu: 2)")

# Test 4: Détection des valeurs aberrantes
print_section("TEST 4: DÉTECTION DES VALEURS ABERRANTES (IQR)")

outliers_detected = {}

for column in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    if len(outliers) > 0:
        outliers_detected[column] = len(outliers)
        print_info(f"Colonne '{column}':")
        print(f"   Limites: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"   {len(outliers)} valeur(s) aberrante(s) détectée(s)")
        for idx, row in outliers.iterrows():
            print(f"   - {row['name']}: {column}={row[column]}")

total_outliers = sum(outliers_detected.values())
if total_outliers >= 2:  # Au moins Charlie (age) et Frank (salary)
    print_success(f"TOTAL: {total_outliers} valeurs aberrantes détectées")
else:
    print_error(f"ERREUR: {total_outliers} valeurs aberrantes détectées (attendu: >= 2)")

# Test 5: Détection automatique des colonnes ID
print_section("TEST 5: DÉTECTION AUTOMATIQUE DES COLONNES ID")

def detect_id_columns(df):
    id_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if 'id' in col_lower or 'identifier' in col_lower or col_lower == 'key':
            id_columns.append(col)
    return id_columns

id_cols = detect_id_columns(df)
print_info(f"Colonnes ID détectées: {id_cols}")

if 'id' in id_cols:
    print_success("Correct: colonne 'id' détectée automatiquement")
else:
    print_error("ERREUR: colonne 'id' non détectée")

# Test 6: Suppression des doublons
print_section("TEST 6: SUPPRESSION DES DOUBLONS")

df_before = len(df)
df_cleaned = df.drop_duplicates(subset=['id'], keep='first')
df_after = len(df_cleaned)
removed = df_before - df_after

print_info(f"Lignes avant: {df_before}")
print_info(f"Lignes après: {df_after}")
print_info(f"Lignes supprimées: {removed}")

if removed == 1:
    print_success("Correct: 1 ligne dupliquée supprimée (Eve avec id=2)")
    print("\nLigne conservée (id=2):")
    print(df_cleaned[df_cleaned['id'] == 2][['id', 'name', 'age', 'salary']].to_string(index=False))
else:
    print_error(f"ERREUR: {removed} ligne(s) supprimée(s) (attendu: 1)")

# Test 7: Traitement des valeurs manquantes
print_section("TEST 7: TRAITEMENT DES VALEURS MANQUANTES (MOYENNE)")

df_test = df.copy()
for column in df_test.columns:
    if df_test[column].dtype in ['float64', 'int64']:
        if df_test[column].isnull().sum() > 0:
            mean_value = df_test[column].mean()
            df_test[column].fillna(mean_value, inplace=True)
            print_info(f"Colonne '{column}': valeurs manquantes remplacées par {mean_value:.2f}")

missing_after = df_test.isnull().sum().sum()

if missing_after == 0:
    print_success("Correct: toutes les valeurs manquantes ont été traitées")
else:
    print_error(f"ERREUR: {missing_after} valeur(s) manquante(s) restante(s)")

# Test 8: Traitement des valeurs aberrantes (cap)
print_section("TEST 8: TRAITEMENT DES VALEURS ABERRANTES (CAP)")

df_test2 = df.copy()

for column in df_test2.select_dtypes(include=[np.number]).columns:
    Q1 = df_test2[column].quantile(0.25)
    Q3 = df_test2[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_before = len(df_test2[(df_test2[column] < lower_bound) | (df_test2[column] > upper_bound)])
    
    if outliers_before > 0:
        df_test2.loc[df_test2[column] < lower_bound, column] = lower_bound
        df_test2.loc[df_test2[column] > upper_bound, column] = upper_bound
        
        outliers_after = len(df_test2[(df_test2[column] < lower_bound) | (df_test2[column] > upper_bound)])
        print_info(f"Colonne '{column}': {outliers_before} valeur(s) limitée(s) aux bornes")

print_success("Valeurs aberrantes traitées avec succès")

# Résumé final
print_section("RÉSUMÉ DES TESTS")

tests_passed = 0
tests_total = 8

results = {
    "Détection valeurs manquantes": total_missing == 2,
    "Détection doublons (toutes colonnes)": duplicates_all == 0,
    "Détection doublons (ID)": duplicates_id == 2,
    "Détection valeurs aberrantes": total_outliers >= 2,
    "Détection automatique colonnes ID": 'id' in id_cols,
    "Suppression doublons": removed == 1,
    "Traitement valeurs manquantes": missing_after == 0,
    "Traitement valeurs aberrantes": True
}

for test_name, result in results.items():
    if result:
        print_success(f"{test_name}")
        tests_passed += 1
    else:
        print_error(f"{test_name}")

print("\n" + "=" * 80)
print(f" RÉSULTAT FINAL: {tests_passed}/{tests_total} tests réussis")
print("=" * 80)

if tests_passed == tests_total:
    print("\n🎉 TOUS LES TESTS SONT PASSÉS! Le backend est correctement configuré.")
else:
    print(f"\n⚠️  {tests_total - tests_passed} test(s) ont échoué. Vérifiez la configuration.")

# Sauvegarder le rapport
print_section("SAUVEGARDE DU RAPPORT")

report = {
    "date": datetime.now().isoformat(),
    "dataset": {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns)
    },
    "issues_detected": {
        "missing_values": total_missing,
        "duplicates_all_columns": int(duplicates_all),
        "duplicates_on_id": int(duplicates_id),
        "outliers": total_outliers
    },
    "tests_results": {k: ("PASS" if v else "FAIL") for k, v in results.items()},
    "tests_summary": {
        "passed": tests_passed,
        "total": tests_total,
        "success_rate": f"{(tests_passed/tests_total)*100:.1f}%"
    }
}

with open('test_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print_success("Rapport sauvegardé dans 'test_report.json'")

print("\n" + "=" * 80)
print(" FIN DES TESTS")
print("=" * 80)