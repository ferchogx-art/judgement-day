#!/usr/bin/env python3
"""
Judgement Day - Voice Assistant
Asistente de voz modular tipo Alexa con arquitectura escalable
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Audio & Speech Recognition
import speech_recognition as sr
import pyttsx3

# LLM API
import google.generativeai as genai

# Web & Weather
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class AudioInput:
    """Módulo de entrada de audio - Escucha continua del micrófono"""
    
    def __init__(self, language: str = 'es-ES'):
        self.recognizer = sr.Recognizer()
        self.language = language
        self.mic = sr.Microphone()
        
    def listen(self) -> Optional[str]:
        """Escucha de forma continua el micrófono y retorna el texto reconocido"""
        try:
            with self.mic as source:
                # Ajustar niveles de ruido ambiental
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("🎤 Escuchando...")
                
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                
            try:
                # Usar Google Speech Recognition (gratuito)
                text = self.recognizer.recognize_google(audio, language=self.language)
                logger.info(f"📝 Reconocido: {text}")
                return text
            except sr.UnknownValueError:
                logger.warning("No se entendió el audio")
                return None
            except sr.RequestError as e:
                logger.error(f"Error en Speech Recognition: {e}")
                return None
                
        except sr.RequestError as e:
            logger.error(f"Error de micrófono: {e}")
            return None


class WeatherService:
    """Módulo de consulta de clima - Integración con OpenWeatherMap"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        
    def get_weather(self, city: str, language: str = 'es') -> str:
        """Obtiene información del clima actual de una ciudad"""
        if not self.api_key:
            logger.warning("OpenWeatherMap API key no configurada")
            return "No tengo acceso a información del clima en este momento."
        
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric',
                'lang': language
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            
            # Extraer información relevante
            temp = data['main']['temp']
            feels_like = data['main']['feels_like']
            description = data['weather'][0]['description']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            
            # Formatear respuesta concisa
            if language == 'es':
                weather_text = (
                    f"En {city}: {temp}°C (se siente como {feels_like}°C). "
                    f"{description.capitalize()}. Humedad: {humidity}%. "
                    f"Viento: {wind_speed} m/s."
                )
            else:
                weather_text = (
                    f"In {city}: {temp}°C (feels like {feels_like}°C). "
                    f"{description.capitalize()}. Humidity: {humidity}%. "
                    f"Wind: {wind_speed} m/s."
                )
            
            logger.info(f"🌤️ Clima obtenido: {weather_text}")
            return weather_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error obteniendo clima: {e}")
            return f"No pude obtener el clima de {city}."
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parseando respuesta de clima: {e}")
            return f"Error procesando datos del clima."


class LLMBrain:
    """Módulo de procesamiento LLM - Cerebro de Judgement Day"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY no configurada en variables de entorno")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.weather_service = WeatherService()
        
    def _detect_language(self, text: str) -> str:
        """Detecta el idioma del texto (simplificado)"""
        spanish_words = ['qué', 'cómo', 'dónde', 'cuándo', 'clima', 'eres', 'hola']
        spanish_count = sum(1 for word in spanish_words if word in text.lower())
        
        return 'es' if spanish_count > 0 else 'en'
    
    def _extract_weather_request(self, text: str) -> Optional[tuple]:
        """Extrae si el usuario pregunta por el clima y la ciudad"""
        weather_keywords_es = ['clima', 'tiempo', 'temperatura', 'lluvia', 'nieve', 'weather', 'forecast']
        weather_keywords_en = ['weather', 'temperature', 'climate', 'rain', 'snow', 'forecast']
        
        text_lower = text.lower()
        
        # Detectar si es una pregunta de clima
        is_weather_question = any(kw in text_lower for kw in weather_keywords_es + weather_keywords_en)
        
        if not is_weather_question:
            return None
        
        # Intentar extraer nombre de ciudad (muy simplificado)
        # En producción, usar NER o regex más sofisticado
        words = text.split()
        for i, word in enumerate(words):
            if any(kw in text_lower[:text_lower.find(word)] for kw in weather_keywords_es + weather_keywords_en):
                if i + 1 < len(words):
                    potential_city = ' '.join(words[i+1:i+3])
                    return ('weather', potential_city)
        
        return ('weather', 'current_location')
    
    def process(self, user_input: str) -> str:
        """Procesa entrada del usuario y retorna respuesta de la IA"""
        try:
            # Detectar idioma
            detected_lang = self._detect_language(user_input)
            language_name = 'español' if detected_lang == 'es' else 'inglés'
            
            logger.info(f"🧠 Idioma detectado: {language_name}")
            
            # Verificar si es una pregunta de clima
            weather_request = self._extract_weather_request(user_input)
            
            if weather_request and weather_request[0] == 'weather':
                city = weather_request[1]
                weather_info = self.weather_service.get_weather(city, detected_lang)
                # Usar el clima como contexto para la respuesta
                context = f"Información del clima: {weather_info}\n\n"
            else:
                context = ""
            
            # System Prompt estricto de Judgement Day
            system_prompt = f"""Tu nombre es Judgement Day. Eres un asistente de voz muy conciso y eficiente.
Identifica de forma automática el idioma del usuario y respóndele EXACTAMENTE en ese mismo idioma.
Si te habla en español responde en español, si te habla en inglés responde en inglés.
Sé conciso, directo y útil. Máximo 2 oraciones por respuesta.
{context}"""
            
            # Llamar a Gemini
            response = self.model.generate_content(
                system_prompt + f"\n\nUsuario: {user_input}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.7
                )
            )
            
            answer = response.text.strip()
            logger.info(f"🤖 Respuesta IA: {answer}")
            return answer
            
        except Exception as e:
            logger.error(f"Error en LLM Brain: {e}")
            return "Lo siento, hubo un error procesando tu solicitud."


class AudioOutput:
    """Módulo de salida de audio - Text-to-Speech"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Velocidad de lectura
        self.engine.setProperty('volume', 0.9)  # Volumen
        
    def speak(self, text: str, language: str = 'es') -> None:
        """Convierte texto a voz y lo reproduce"""
        try:
            # Configurar idioma (muy básico - en producción usar voces específicas)
            voices = self.engine.getProperty('voices')
            
            if language == 'es' and len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
            else:
                self.engine.setProperty('voice', voices[0].id)
            
            logger.info("🔊 Reproduciendo audio...")
            self.engine.say(text)
            self.engine.runAndWait()
            
        except Exception as e:
            logger.error(f"Error en Text-to-Speech: {e}")


class JudgementDay:
    """Orquestador principal - Asistente de voz Judgement Day"""
    
    def __init__(self):
        """Inicializa todos los módulos del asistente"""
        try:
            self.audio_input = AudioInput(language='es-ES')
            self.llm_brain = LLMBrain()
            self.audio_output = AudioOutput()
            self.is_running = False
            
            logger.info("✅ Judgement Day inicializado correctamente")
        except Exception as e:
            logger.error(f"❌ Error iniciando Judgement Day: {e}")
            sys.exit(1)
    
    def run_continuous(self) -> None:
        """Ejecuta el asistente en modo escucha continua"""
        self.is_running = True
        logger.info("\n" + "="*60)
        logger.info("🤖 JUDGEMENT DAY - Voice Assistant Iniciado")
        logger.info("Escuchando... (di 'salir' para terminar)")
        logger.info("="*60 + "\n")
        
        try:
            while self.is_running:
                # 1. Escuchar entrada de audio
                user_input = self.audio_input.listen()
                
                if not user_input:
                    continue
                
                # Verificar comando de salida
                if any(word in user_input.lower() for word in ['salir', 'exit', 'quit', 'adiós', 'adios']):
                    logger.info("👋 Terminando Judgement Day...")
                    self.audio_output.speak("Adiós. Hasta pronto.", language='es')
                    self.is_running = False
                    break
                
                # 2. Procesar con LLM
                response = self.llm_brain.process(user_input)
                
                # 3. Reproducir respuesta en audio
                detected_lang = self.llm_brain._detect_language(user_input)
                self.audio_output.speak(response, language=detected_lang)
                
                print("\n" + "-"*60 + "\n")
        
        except KeyboardInterrupt:
            logger.info("\n⚠️ Interrupción del usuario")
            self.audio_output.speak("Assistente detenido.", language='es')
        except Exception as e:
            logger.error(f"❌ Error crítico: {e}")
            sys.exit(1)
    
    def run_single(self, text: str) -> None:
        """Ejecuta el asistente con una única entrada de texto (modo prueba)"""
        logger.info(f"📝 Entrada: {text}")
        response = self.llm_brain.process(text)
        logger.info(f"🤖 Respuesta: {response}")
        
        detected_lang = self.llm_brain._detect_language(text)
        self.audio_output.speak(response, language=detected_lang)


def main():
    """Punto de entrada principal"""
    print("\n╔════════════════════════════════════════════════════╗")
    print("║         JUDGEMENT DAY - Voice Assistant           ║")
    print("║           Powered by Google Gemini                ║")
    print("╚════════════════════════════════════════════════════╝\n")
    
    # Crear instancia de Judgement Day
    jd = JudgementDay()
    
    # Modo de ejecución
    if len(sys.argv) > 1:
        # Modo prueba: python main.py "tu pregunta aquí"
        test_input = ' '.join(sys.argv[1:])
        jd.run_single(test_input)
    else:
        # Modo continuo: escucha infinita
        jd.run_continuous()


if __name__ == '__main__':
    main()
