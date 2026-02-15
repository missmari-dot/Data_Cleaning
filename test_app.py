import unittest
import os
import sys
import pandas as pd
import json
from io import BytesIO

# Ajouter le chemin du dossier parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, handle_missing_values, handle_outliers, handle_duplicates, normalize_data

class TestDataProcessingAPI(unittest.TestCase):
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Créer un DataFrame de test
        self.test_data = pd.DataFrame({
            'ID': [1, 2, 3, 4, 1],
            'Age': [25, None, 30, 35, 25],
            'Salary': [50000, 60000, 999999, 75000, 50000],
            'Department': ['IT', 'HR', 'Finance', 'IT', 'IT']
        })
    
    def test_health_check(self):
        """Test de l'endpoint health check"""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
    
    def test_handle_missing_values_mean(self):
        """Test du traitement des valeurs manquantes avec la moyenne"""
        df = self.test_data.copy()
        df_clean, report = handle_missing_values(df, strategy='mean')
        
        # Vérifier qu'il n'y a plus de valeurs manquantes
        self.assertEqual(df_clean['Age'].isnull().sum(), 0)
        
        # Vérifier que la valeur a été remplacée par la moyenne
        expected_mean = (25 + 30 + 35 + 25) / 4
        self.assertEqual(df_clean.loc[1, 'Age'], expected_mean)
        
        # Vérifier le rapport
        self.assertEqual(report['total_missing'], 1)
    
    def test_handle_missing_values_median(self):
        """Test du traitement des valeurs manquantes avec la médiane"""
        df = self.test_data.copy()
        df_clean, report = handle_missing_values(df, strategy='median')
        
        self.assertEqual(df_clean['Age'].isnull().sum(), 0)
        self.assertEqual(report['total_missing'], 1)
    
    def test_handle_missing_values_mode(self):
        """Test du traitement des valeurs manquantes avec le mode"""
        df = self.test_data.copy()
        df_clean, report = handle_missing_values(df, strategy='mode')
        
        self.assertEqual(df_clean['Age'].isnull().sum(), 0)
        self.assertEqual(report['total_missing'], 1)
    
    def test_handle_missing_values_drop(self):
        """Test du traitement des valeurs manquantes avec suppression"""
        df = self.test_data.copy()
        original_rows = len(df)
        df_clean, report = handle_missing_values(df, strategy='drop')
        
        # Vérifier qu'une ligne a été supprimée
        self.assertEqual(len(df_clean), original_rows - 1)
        self.assertEqual(df_clean['Age'].isnull().sum(), 0)
    
    def test_handle_outliers_iqr_cap(self):
        """Test du traitement des outliers avec IQR et cap"""
        df = self.test_data.copy()
        df_clean, report = handle_outliers(df, method='iqr', action='cap')
        
        # Vérifier que l'outlier a été traité
        self.assertTrue(df_clean['Salary'].max() < 999999)
        self.assertGreater(report['total_outliers'], 0)
    
    def test_handle_outliers_iqr_remove(self):
        """Test du traitement des outliers avec IQR et remove"""
        df = self.test_data.copy()
        original_rows = len(df)
        df_clean, report = handle_outliers(df, method='iqr', action='remove')
        
        # Vérifier que des lignes ont été supprimées
        self.assertLess(len(df_clean), original_rows)
    
    def test_handle_outliers_zscore(self):
        """Test du traitement des outliers avec Z-score"""
        df = self.test_data.copy()
        df_clean, report = handle_outliers(df, method='zscore', action='cap')
        
        # Vérifier qu'un rapport est généré
        self.assertIn('total_outliers', report)
    
    def test_handle_duplicates(self):
        """Test du traitement des doublons"""
        df = self.test_data.copy()
        original_rows = len(df)
        df_clean, report = handle_duplicates(df)
        
        # Vérifier qu'un doublon a été supprimé
        self.assertEqual(len(df_clean), original_rows - 1)
        self.assertEqual(report['duplicates_found'], 1)
        self.assertEqual(report['duplicates_removed'], 1)
    
    def test_normalize_data_standard(self):
        """Test de la normalisation avec standardisation"""
        df = self.test_data.copy()
        # Supprimer les doublons pour faciliter le test
        df = df.drop_duplicates()
        df_norm, report = normalize_data(df, method='standard')
        
        # Vérifier que les colonnes numériques ont été normalisées
        self.assertIn('ID', report['normalized_columns'])
        self.assertIn('Salary', report['normalized_columns'])
        
        # Vérifier que la moyenne est proche de 0 (avec tolérance)
        self.assertAlmostEqual(df_norm['Age'].mean(), 0, places=5)
    
    def test_normalize_data_minmax(self):
        """Test de la normalisation avec Min-Max"""
        df = self.test_data.copy()
        df = df.drop_duplicates()
        df_norm, report = normalize_data(df, method='minmax')
        
        # Vérifier que les valeurs sont entre 0 et 1
        self.assertTrue((df_norm['Age'] >= 0).all())
        self.assertTrue((df_norm['Age'] <= 1).all())
        self.assertEqual(report['method'], 'minmax')
    
    def test_analyze_endpoint(self):
        """Test de l'endpoint d'analyse"""
        # Créer un fichier CSV de test
        csv_content = self.test_data.to_csv(index=False)
        data = {
            'file': (BytesIO(csv_content.encode()), 'test.csv')
        }
        
        response = self.app.post('/api/analyze', 
                                data=data,
                                content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        # Vérifier les données retournées
        self.assertEqual(result['rows'], 5)
        self.assertEqual(result['columns'], 4)
        self.assertIn('missing_values', result)
        self.assertIn('duplicates', result)
    
    def test_analyze_endpoint_invalid_file(self):
        """Test de l'endpoint d'analyse avec un fichier invalide"""
        data = {
            'file': (BytesIO(b'invalid content'), 'test.txt')
        }
        
        response = self.app.post('/api/analyze', 
                                data=data,
                                content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 400)
    
    def test_process_endpoint(self):
        """Test de l'endpoint de traitement"""
        # Créer un fichier CSV de test
        csv_content = self.test_data.to_csv(index=False)
        data = {
            'file': (BytesIO(csv_content.encode()), 'test.csv'),
            'missing_strategy': 'mean',
            'outlier_method': 'iqr',
            'outlier_action': 'cap',
            'normalization_method': 'standard',
            'output_format': 'csv'
        }
        
        response = self.app.post('/api/process', 
                                data=data,
                                content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data)
        
        # Vérifier les données retournées
        self.assertTrue(result['success'])
        self.assertIn('report', result)
        self.assertIn('download_url', result)
    
    def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer les fichiers temporaires créés lors des tests
        uploads_dir = 'uploads'
        processed_dir = 'processed'
        
        if os.path.exists(uploads_dir):
            for file in os.listdir(uploads_dir):
                os.remove(os.path.join(uploads_dir, file))
        
        if os.path.exists(processed_dir):
            for file in os.listdir(processed_dir):
                os.remove(os.path.join(processed_dir, file))

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('processed', exist_ok=True)
    
    # Lancer les tests
    unittest.main(verbosity=2)
