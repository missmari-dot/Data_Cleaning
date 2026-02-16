from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from werkzeug.utils import secure_filename
import json
import xml.etree.ElementTree as ET
from datetime import datetime
import io

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'json', 'xml'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


def allowed_file(filename):
    """Vérifie si le fichier a une extension autorisée"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_file(filepath):
    """Lit un fichier et retourne un DataFrame pandas"""
    extension = filepath.rsplit('.', 1)[1].lower()
    
    try:
        if extension == 'csv':
            return pd.read_csv(filepath)
        elif extension in ['xlsx', 'xls']:
            return pd.read_excel(filepath)
        elif extension == 'json':
            return pd.read_json(filepath)
        elif extension == 'xml':
            tree = ET.parse(filepath)
            root = tree.getroot()
            data = []
            for child in root:
                row = {}
                for subchild in child:
                    row[subchild.tag] = subchild.text
                data.append(row)
            return pd.DataFrame(data)
    except Exception as e:
        raise Exception(f"Erreur lors de la lecture du fichier: {str(e)}")


def detect_id_columns(df):
    """Détecte automatiquement les colonnes qui pourraient être des identifiants"""
    id_columns = []
    for col in df.columns:
        col_lower = col.lower()
        # Chercher des colonnes avec 'id' dans le nom
        if 'id' in col_lower or 'identifier' in col_lower or col_lower == 'key':
            id_columns.append(col)
    return id_columns


def detect_outliers(df, column, method='iqr'):
    """Détecte les valeurs aberrantes dans une colonne"""
    outliers = []
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3].index.tolist()
    
    return outliers


def handle_missing_values(df, strategy='mean'):
    """Traite les valeurs manquantes"""
    report = {
        'columns_with_missing': {},
        'total_missing': 0
    }
    
    for column in df.columns:
        missing_count = df[column].isnull().sum()
        if missing_count > 0:
            report['columns_with_missing'][column] = int(missing_count)
            report['total_missing'] += int(missing_count)
            
            if df[column].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    df[column].fillna(df[column].mean(), inplace=True)
                elif strategy == 'median':
                    df[column].fillna(df[column].median(), inplace=True)
                elif strategy == 'mode':
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 0, inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
            else:
                if strategy == 'mode':
                    df[column].fillna(df[column].mode()[0] if not df[column].mode().empty else 'Unknown', inplace=True)
                elif strategy == 'drop':
                    df.dropna(subset=[column], inplace=True)
                else:
                    df[column].fillna('Unknown', inplace=True)
    
    return df, report


def handle_outliers(df, method='iqr', action='cap'):
    """Traite les valeurs aberrantes"""
    report = {
        'columns_with_outliers': {},
        'total_outliers': 0
    }
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for column in numeric_columns:
        outlier_indices = detect_outliers(df, column, method)
        
        if len(outlier_indices) > 0:
            report['columns_with_outliers'][column] = len(outlier_indices)
            report['total_outliers'] += len(outlier_indices)
            
            if action == 'remove':
                df = df.drop(outlier_indices)
            elif action == 'cap':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound
    
    return df, report


def handle_duplicates(df, duplicate_columns=None):
    """
    Traite les valeurs dupliquées avec option flexible
    
    Args:
        df: DataFrame à traiter
        duplicate_columns: Liste des colonnes à vérifier pour les doublons
                          None = toutes les colonnes (comportement par défaut)
                          ['id'] = vérifier uniquement la colonne 'id'
                          ['id', 'email'] = vérifier les colonnes 'id' et 'email'
    """
    # Déterminer les colonnes à vérifier
    if duplicate_columns is None or duplicate_columns == 'all':
        subset_cols = None  # Toutes les colonnes
        detection_method = "toutes les colonnes"
    else:
        # Vérifier que les colonnes existent
        subset_cols = [col for col in duplicate_columns if col in df.columns]
        detection_method = f"colonnes: {', '.join(subset_cols)}"
    
    # Compter tous les doublons
    duplicates_mask = df.duplicated(subset=subset_cols, keep=False)
    duplicates_count = duplicates_mask.sum()
    
    # Supprimer les doublons en gardant la première occurrence
    df_before = len(df)
    df = df.drop_duplicates(subset=subset_cols, keep='first')
    df_after = len(df)
    
    report = {
        'duplicates_found': int(duplicates_count),
        'duplicates_removed': int(df_before - df_after),
        'detection_method': detection_method,
        'columns_checked': subset_cols if subset_cols else 'all'
    }
    
    return df, report


def normalize_data(df, method='standard'):
    """Normalise les données numériques"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        return df, {'normalized_columns': [], 'method': method}
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        return df, {'normalized_columns': [], 'method': method}
    
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    report = {
        'normalized_columns': list(numeric_columns),
        'method': method
    }
    
    return df, report


@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérifie l'état de l'API"""
    return jsonify({'status': 'ok', 'message': 'API is running'})


@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """Analyse un fichier sans le traiter - VERSION AMÉLIORÉE"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = read_file(filepath)
        
        # Détection automatique des colonnes ID
        id_columns = detect_id_columns(df)
        
        # Analyse améliorée des valeurs manquantes
        missing_values = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_values[column] = int(missing_count)
        
        # Détection des doublons sur TOUTES les colonnes
        duplicates_all = int(df.duplicated(keep=False).sum())
        
        # Détection des doublons sur colonnes ID (si détectées)
        duplicates_id = 0
        if id_columns:
            duplicates_id = int(df.duplicated(subset=id_columns, keep=False).sum())
        
        # Analyse des données
        analysis = {
            'filename': filename,
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'column_types': df.dtypes.astype(str).to_dict(),
            'missing_values': missing_values,
            'duplicates': duplicates_all,  # Doublons complets
            'duplicates_on_id': duplicates_id,  # Doublons sur ID
            'id_columns': id_columns,  # Colonnes ID détectées
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'preview': df.head(5).to_dict('records')
        }
        
        # Détection des valeurs aberrantes
        outliers_info = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            outlier_indices = detect_outliers(df, column)
            if len(outlier_indices) > 0:
                outliers_info[column] = len(outlier_indices)
        
        analysis['outliers'] = outliers_info
        
        # Nettoyage du fichier temporaire
        os.remove(filepath)
        
        return jsonify(analysis)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/process', methods=['POST'])
def process_file():
    """Traite un fichier selon les paramètres fournis - VERSION AMÉLIORÉE"""
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier fourni'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé'}), 400
    
    try:
        # Récupération des paramètres
        missing_strategy = request.form.get('missing_strategy', 'mean')
        outlier_method = request.form.get('outlier_method', 'iqr')
        outlier_action = request.form.get('outlier_action', 'cap')
        normalization_method = request.form.get('normalization_method', 'standard')
        output_format = request.form.get('output_format', 'csv')
        
        # NOUVEAU: Paramètre pour la détection des doublons
        duplicate_detection = request.form.get('duplicate_detection', 'all')
        duplicate_columns_str = request.form.get('duplicate_columns', '')
        
        # Sauvegarde du fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Lecture du fichier
        df = read_file(filepath)
        original_rows = len(df)
        
        # Détection automatique des colonnes ID si nécessaire
        if duplicate_detection == 'id':
            id_columns = detect_id_columns(df)
            duplicate_columns = id_columns if id_columns else None
        elif duplicate_detection == 'custom' and duplicate_columns_str:
            duplicate_columns = [col.strip() for col in duplicate_columns_str.split(',')]
        else:
            duplicate_columns = None
        
        # Rapport de traitement
        processing_report = {
            'original_file': filename,
            'original_rows': original_rows,
            'original_columns': len(df.columns),
            'steps': []
        }
        
        # 1. Traitement des valeurs manquantes
        df, missing_report = handle_missing_values(df, missing_strategy)
        processing_report['steps'].append({
            'step': 'Traitement des valeurs manquantes',
            'details': missing_report
        })
        
        # 2. Traitement des valeurs aberrantes
        df, outliers_report = handle_outliers(df, outlier_method, outlier_action)
        processing_report['steps'].append({
            'step': 'Traitement des valeurs aberrantes',
            'details': outliers_report
        })
        
        # 3. Traitement des doublons - VERSION FLEXIBLE
        df, duplicates_report = handle_duplicates(df, duplicate_columns)
        processing_report['steps'].append({
            'step': 'Traitement des doublons',
            'details': duplicates_report
        })
        
        # 4. Normalisation
        df, normalization_report = normalize_data(df, normalization_method)
        processing_report['steps'].append({
            'step': 'Normalisation des données',
            'details': normalization_report
        })
        
        # Statistiques finales
        processing_report['final_rows'] = len(df)
        processing_report['final_columns'] = len(df.columns)
        processing_report['rows_removed'] = original_rows - len(df)
        
        # Sauvegarde du fichier traité
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(filename)[0]
        
        if output_format == 'csv':
            output_filename = f"{base_name}_processed_{timestamp}.csv"
            output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            df.to_csv(output_filepath, index=False)
        elif output_format == 'excel':
            output_filename = f"{base_name}_processed_{timestamp}.xlsx"
            output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            df.to_excel(output_filepath, index=False)
        elif output_format == 'json':
            output_filename = f"{base_name}_processed_{timestamp}.json"
            output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            df.to_json(output_filepath, orient='records', indent=2)
        
        processing_report['output_file'] = output_filename
        
        # Nettoyage du fichier temporaire
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'report': processing_report,
            'download_url': f'/api/download/{output_filename}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Télécharge un fichier traité"""
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'Fichier non trouvé'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run()
