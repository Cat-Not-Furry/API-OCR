import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
from PIL import Image
import os
from pathlib Path
import logging
import json
from typing import Dict, List, Tuple
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRPerfeccionado:
    def __init__(self):
        BASE_DIR = Path(__file__).parent.absolute()
        self.tesseract_path = os.path.join(BASE_DIR, "bin", "tesseract")
        self.tessdata_path = os.path.join(BASE_DIR, "tessdata")
        
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        os.environ['TESSDATA_PREFIX'] = self.tessdata_path
        
        # Configuración específica para formularios
        self.config_impreso = r'--oem 3 --psm 6 -l spa --tessedit_char_whitelist=abcdefghijklmnñopqrstuvwxyzABCDEFGHIJKLMNÑOPQRSTUVWXYZáéíóúÁÉÍÓÚ0123456789.,;:¡!¿?()[]-'
        self.config_checkbox = r'--oem 3 --psm 8 -l spa --tessedit_char_whitelist=OX√•'
    
    def preprocesar_imagen(self, image_bytes: bytes) -> np.ndarray:
        """Pipeline completo de preprocesamiento para fotos de celular"""
        # Convertir bytes a imagen
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Redimensionar si es muy pequeña
        h, w = img.shape[:2]
        if min(h, w) < 1000:
            scale = 2000 / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 2. Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Corrección de inclinación (skew)
        coords = np.column_stack(np.where(gray > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h, w) = gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # 4. Eliminar sombras con CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 5. Reducción de ruido
        gray = cv2.medianBlur(gray, 3)
        
        # 6. Binarización adaptativa (clave para fotos con iluminación desigual)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def detectar_checkboxes(self, img: np.ndarray) -> List[Dict]:
        """Detecta checkboxes [ ], O, (a) y su estado"""
        checkboxes = []
        
        # Umbral para encontrar contornos
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 5000:  # Filtrar por tamaño
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Detectar cuadrados [ ]
            if 0.8 < aspect_ratio < 1.2 and area > 100:
                # Extraer ROI del checkbox
                roi = img[y:y+h, x:x+w]
                
                # Detectar si está marcado (presencia de píxeles oscuros en el interior)
                roi_mean = cv2.mean(roi)[0]
                marcado = roi_mean < 200  # Umbral simple
                
                # Verificar con OCR para X o ✓
                texto = pytesseract.image_to_string(roi, config=self.config_checkbox).strip()
                if texto in ['X', 'x', '√', '•']:
                    marcado = True
                
                checkboxes.append({
                    'tipo': 'cuadrado',
                    'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                    'marcado': marcado,
                    'confianza': float(100 - roi_mean/2.55)  # Normalizado a 0-100
                })
            
            # Detectar círculos O
            elif len(contour) > 6 and area > 100:  # Aproximadamente circular
                # Calcular circularidad
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity > 0.7:  # Es un círculo
                    roi = img[y:y+h, x:x+w]
                    roi_mean = cv2.mean(roi)[0]
                    marcado = roi_mean < 200
                    
                    checkboxes.append({
                        'tipo': 'circulo',
                        'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                        'marcado': marcado,
                        'confianza': float(100 - roi_mean/2.55)
                    })
            
            # Detectar incisos (a), (A)
            elif aspect_ratio > 1.5 and area < 500:
                roi = img[y:y+h, x:x+w]
                texto = pytesseract.image_to_string(roi, config=r'--psm 8 -l spa').strip()
                if re.match(r'^\([a-zA-Z]\)$', texto):
                    checkboxes.append({
                        'tipo': 'inciso',
                        'texto': texto,
                        'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                        'marcado': False  # Los incisos no se marcan, son etiquetas
                    })
        
        return checkboxes
    
    def detectar_campos_texto(self, img: np.ndarray) -> List[Dict]:
        """Detecta líneas _____ para campos de texto"""
        campos = []
        
        # Detectar líneas horizontales largas
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                # Es casi horizontal
                if abs(y2 - y1) < 10 and length > 100:
                    campos.append({
                        'tipo': 'campo_texto',
                        'x1': int(x1), 'y1': int(y1),
                        'x2': int(x2), 'y2': int(y2),
                        'longitud': int(length)
                    })
        
        return campos
    
    def extraer_texto_estructurado(self, img: np.ndarray) -> Dict:
        """Extrae todo el texto manteniendo estructura"""
        # OCR con hOCR para obtener posiciones
        config = r'--psm 6 -l spa --oem 3 hocr'
        hocr = pytesseract.image_to_pdf_or_hocr(img, extension='hocr', config=config)
        
        # Parsear hOCR (simplificado)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(hocr.decode('utf-8'))
        
        parrafos = []
        lineas = []
        
        for line in root.findall('.//span[@class="ocr_line"]'):
            line_bbox = line.get('title', '').split('bbox')[1].strip().split()
            line_text = []
            
            for word in line.findall('.//span[@class="ocrx_word"]'):
                word_text = word.text or ''
                if word_text.strip():
                    line_text.append(word_text)
            
            if line_text:
                lineas.append({
                    'texto': ' '.join(line_text),
                    'x1': int(line_bbox[0]), 'y1': int(line_bbox[1]),
                    'x2': int(line_bbox[2]), 'y2': int(line_bbox[3])
                })
        
        return {'parrafos': parrafos, 'lineas': lineas}
    
    def asociar_checkboxes_con_texto(self, checkboxes: List[Dict], lineas: List[Dict]) -> List[Dict]:
        """Asocia cada checkbox con el texto más cercano (la pregunta)"""
        for checkbox in checkboxes:
            # Encontrar la línea más cercana arriba del checkbox
            checkbox_center_y = checkbox['y'] + checkbox['h']//2
            lineas_cercanas = []
            
            for linea in lineas:
                # La línea debe estar arriba del checkbox
                if linea['y2'] < checkbox['y']:
                    distancia = checkbox['y'] - linea['y2']
                    if distancia < 100:  # Máximo 100 píxels de distancia
                        lineas_cercanas.append({
                            'texto': linea['texto'],
                            'distancia': distancia
                        })
            
            # Ordenar por distancia
            lineas_cercanas.sort(key=lambda x: x['distancia'])
            
            if lineas_cercanas:
                checkbox['pregunta_asociada'] = lineas_cercanas[0]['texto']
                checkbox['distancia_pregunta'] = lineas_cercanas[0]['distancia']
            else:
                checkbox['pregunta_asociada'] = None
        
        return checkboxes
    
    def procesar_formulario(self, image_bytes: bytes) -> Dict:
        """Pipeline completo para formularios educativos"""
        # 1. Preprocesar
        img_procesada = self.preprocesar_imagen(image_bytes)
        
        # 2. Detectar checkboxes
        checkboxes = self.detectar_checkboxes(img_procesada)
        logger.info(f"✅ Detectados {len(checkboxes)} checkboxes")
        
        # 3. Detectar campos de texto
        campos_texto = self.detectar_campos_texto(img_procesada)
        logger.info(f"✅ Detectados {len(campos_texto)} campos de texto")
        
        # 4. Extraer texto estructurado
        texto_estructurado = self.extraer_texto_estructurado(img_procesada)
        logger.info(f"✅ Extraídas {len(texto_estructurado['lineas'])} líneas de texto")
        
        # 5. Asociar checkboxes con preguntas
        checkboxes = self.asociar_checkboxes_con_texto(checkboxes, texto_estructurado['lineas'])
        
        # 6. OCR completo del documento
        texto_completo = pytesseract.image_to_string(img_procesada, config=self.config_impreso)
        
        return {
            'texto_completo': texto_completo,
            'checkboxes': checkboxes,
            'campos_texto': campos_texto,
            'lineas_texto': texto_estructurado['lineas'][:50],  # Limitar para no saturar
            'metadata': {
                'total_checkboxes': len(checkboxes),
                'total_campos': len(campos_texto),
                'total_lineas': len(texto_estructurado['lineas'])
            }
        }

# Integrar con FastAPI
ocr_processor = OCRPerfeccionado()

@app.post("/ocr/perfeccionado")
async def ocr_perfeccionado(file: UploadFile = File(...)):
    """Endpoint con pipeline optimizado para formularios educativos"""
    try:
        contents = await file.read()
        resultado = ocr_processor.procesar_formulario(contents)
        return JSONResponse({
            "success": True,
            "filename": file.filename,
            "resultado": resultado
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(500, f"Error procesando: {str(e)}")
