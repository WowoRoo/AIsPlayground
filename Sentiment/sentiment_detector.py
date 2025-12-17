#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detektor sentymentu rozmówców w języku polskim
Używa modelu opartego na architekturze BART/BERT do analizy sentymentu
"""

import re
import json
import time
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch


class SentimentDetector:
    """Klasa do detekcji sentymentu w rozmowach po polsku"""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"):
        """
        Inicjalizacja detektora sentymentu
        
        Args:
            model_name: Nazwa modelu z Hugging Face. Domyślnie używa wielojęzycznego
                       modelu RoBERTa, który dobrze działa z polskim.
                       Alternatywnie można użyć: 'dkleczek/bert-base-polish-cased-v1'
        """
        print(f"Ładowanie modelu: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używane urządzenie: {self.device}")
        
        # Tworzenie pipeline do analizy sentymentu
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        print("Model załadowany pomyślnie!")
    
    def parse_conversation(self, conversation_text: str) -> List[Dict[str, str]]:
        """
        Parsuje tekst rozmowy na listę wypowiedzi z identyfikacją rozmówcy
        
        Args:
            conversation_text: Tekst rozmowy w formacie:
                              "Rozmówca1: tekst wypowiedzi\nRozmówca2: tekst wypowiedzi"
                              Może zawierać ramy czasowe na początku linii: "[timedate - timedate]"
        
        Returns:
            Lista słowników z kluczami 'speaker' i 'text'
        """
        lines = conversation_text.strip().split('\n')
        messages = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Usuwanie ram czasowych na początku linii w formacie [timedate - timedate]
            # Przykład: [2024-01-01 10:00:00 - 2024-01-01 10:00:05] Anna: Cześć!
            line = re.sub(r'^\[[^\]]+\]\s*', '', line)
            
            # Szukamy wzorca "Nazwa: tekst" lub "NAZWA: tekst"
            match = re.match(r'^([^:]+):\s*(.+)$', line)
            if match:
                speaker = match.group(1).strip()
                text = match.group(2).strip()
                if text:  # Tylko jeśli jest tekst
                    messages.append({
                        'speaker': speaker,
                        'text': text
                    })
            else:
                # Jeśli nie ma wzorca, traktujemy całą linię jako tekst
                # i przypisujemy do ostatniego rozmówcy lub "Nieznany"
                if messages:
                    messages[-1]['text'] += ' ' + line
                else:
                    messages.append({
                        'speaker': 'Nieznany',
                        'text': line
                    })
        
        return messages
    
    def _normalize_sentiment_label(self, label: str) -> str:
        """
        Normalizuje etykietę sentymentu do polskiego formatu
        
        Args:
            label: Etykieta z modelu
        
        Returns:
            Znormalizowana etykieta (POZYTYWNY, NEGATYWNY, NEUTRALNY)
        """
        label_upper = label.upper()
        if 'POSITIVE' in label_upper or 'POZYTYWNY' in label_upper:
            return 'POZYTYWNY'
        elif 'NEGATIVE' in label_upper or 'NEGATYWNY' in label_upper:
            return 'NEGATYWNY'
        else:
            return 'NEUTRALNY'
    
    def analyze_sentiment(self, text: str) -> Dict[str, any]:
        """
        Analizuje sentyment pojedynczego tekstu
        
        Args:
            text: Tekst do analizy
        
        Returns:
            Słownik z wynikami analizy sentymentu
        """
        if not text or not text.strip():
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'text': text
            }
        
        # Ograniczenie długości tekstu (modele mają limity tokenów)
        max_length = 512
        if len(text) > max_length:
            text = text[:max_length]
        
        try:
            result = self.sentiment_pipeline(text)[0]
            
            # Mapowanie etykiet na bardziej czytelne formaty
            label = result['label']
            score = result['score']
            sentiment = self._normalize_sentiment_label(label)
            
            return {
                'label': sentiment,
                'score': score,
                'text': text
            }
        except Exception as e:
            print(f"Błąd podczas analizy sentymentu: {e}")
            return {
                'label': 'NEUTRALNY',
                'score': 0.5,
                'text': text,
                'error': str(e)
            }
    
    def analyze_sentiment_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, any]]:
        """
        Analizuje sentyment wielu tekstów jednocześnie (batch processing)
        
        Args:
            texts: Lista tekstów do analizy
            batch_size: Rozmiar batcha (domyślnie 32)
        
        Returns:
            Lista słowników z wynikami analizy sentymentu
        """
        if not texts:
            return []
        
        # Przygotowanie tekstów (obcięcie do max_length)
        max_length = 512
        processed_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                processed_texts.append("")
                original_indices.append(i)
            else:
                processed_text = text[:max_length] if len(text) > max_length else text
                processed_texts.append(processed_text)
                original_indices.append(i)
        
        results = []
        
        try:
            # Przetwarzanie w batchach
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_results = self.sentiment_pipeline(batch)
                
                # Normalizacja wyników
                for j, result in enumerate(batch_results):
                    label = result['label']
                    score = result['score']
                    sentiment = self._normalize_sentiment_label(label)
                    
                    original_idx = original_indices[i + j]
                    original_text = texts[original_idx]
                    
                    results.append({
                        'label': sentiment,
                        'score': score,
                        'text': original_text[:max_length] if len(original_text) > max_length else original_text
                    })
        
        except Exception as e:
            print(f"Błąd podczas batch analizy sentymentu: {e}")
            # Fallback do pojedynczych analiz w przypadku błędu
            for text in texts:
                results.append(self.analyze_sentiment(text))
        
        return results
    
    def analyze_conversation(self, conversation_text: str) -> Dict[str, any]:
        """
        Analizuje całą rozmowę i zwraca sentyment dla każdego rozmówcy
        
        Args:
            conversation_text: Tekst rozmowy
        
        Returns:
            Słownik z analizą sentymentu dla każdego rozmówcy
        """
        messages = self.parse_conversation(conversation_text)
        
        if not messages:
            return {
                'error': 'Brak wiadomości w rozmowie',
                'messages': []
            }
        
        results = []
        speaker_sentiments = {}
        
        # Rozpoczęcie pomiaru czasu analizy sentymentu
        analysis_start_time = time.time()
        
        # Zbieranie wszystkich tekstów do batch processing
        texts = [msg['text'] for msg in messages]
        
        # Analiza sentymentu w batchach (bardziej wydajne na GPU)
        sentiment_results = self.analyze_sentiment_batch(texts, batch_size=32)
        
        # Mapowanie wyników z powrotem do wiadomości
        for i, msg in enumerate(messages):
            speaker = msg['speaker']
            text = msg['text']
            sentiment_result = sentiment_results[i]
            
            result_entry = {
                'speaker': speaker,
                'text': text,
                'sentiment': sentiment_result['label'],
                'confidence': sentiment_result['score']
            }
            
            results.append(result_entry)
            
            # Agregacja sentymentów dla każdego rozmówcy
            if speaker not in speaker_sentiments:
                speaker_sentiments[speaker] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'total': 0,
                    'avg_confidence': 0.0
                }
            
            speaker_sentiments[speaker]['total'] += 1
            speaker_sentiments[speaker]['avg_confidence'] += sentiment_result['score']
            
            if sentiment_result['label'] == 'POZYTYWNY':
                speaker_sentiments[speaker]['positive'] += 1
            elif sentiment_result['label'] == 'NEGATYWNY':
                speaker_sentiments[speaker]['negative'] += 1
            else:
                speaker_sentiments[speaker]['neutral'] += 1
        
        # Obliczanie średniej pewności dla każdego rozmówcy
        for speaker in speaker_sentiments:
            if speaker_sentiments[speaker]['total'] > 0:
                speaker_sentiments[speaker]['avg_confidence'] /= speaker_sentiments[speaker]['total']
        
        # Określenie dominującego sentymentu dla każdego rozmówcy
        speaker_summary = {}
        for speaker, stats in speaker_sentiments.items():
            total = stats['total']
            if stats['positive'] > stats['negative'] and stats['positive'] > stats['neutral']:
                dominant = 'POZYTYWNY'
            elif stats['negative'] > stats['positive'] and stats['negative'] > stats['neutral']:
                dominant = 'NEGATYWNY'
            else:
                dominant = 'NEUTRALNY'
            
            speaker_summary[speaker] = {
                'dominant_sentiment': dominant,
                'statistics': stats
            }
        
        # Zakończenie pomiaru czasu analizy sentymentu
        analysis_end_time = time.time()
        analysis_time = analysis_end_time - analysis_start_time
        
        return {
            'messages': results,
            'speaker_summary': speaker_summary,
            'total_messages': len(results),
            'analysis_time_seconds': analysis_time
        }
    
    def analyze_from_file(self, file_path: str) -> Dict[str, any]:
        """
        Analizuje rozmowę z pliku tekstowego
        
        Args:
            file_path: Ścieżka do pliku z rozmową
        
        Returns:
            Wyniki analizy sentymentu
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_text = f.read()
            return self.analyze_conversation(conversation_text)
        except Exception as e:
            return {
                'error': f'Błąd podczas czytania pliku: {str(e)}'
            }
    
    def print_results(self, results: Dict[str, any], verbose: bool = False):
        """
        Wyświetla wyniki analizy w czytelnej formie
        
        Args:
            results: Wyniki z metody analyze_conversation
            verbose: Jeśli True, wyświetla szczegółową analizę każdej wiadomości
        """
        if 'error' in results:
            print(f"Błąd: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("ANALIZA SENTYMENTU ROZMOWY")
        print("="*80)
        
        print(f"\nŁączna liczba wiadomości: {results['total_messages']}")
        
        # Wyświetlanie czasu analizy
        if 'analysis_time_seconds' in results:
            analysis_time = results['analysis_time_seconds']
            if analysis_time < 1:
                print(f"Czas analizy sentymentu: {analysis_time * 1000:.2f} ms")
            else:
                print(f"Czas analizy sentymentu: {analysis_time:.2f} s")
            print(f"Średni czas na wiadomość: {analysis_time / results['total_messages'] * 1000:.2f} ms")
        print()
        
        # Szczegółowa analiza wiadomości - tylko jeśli verbose=True
        if verbose:
            print("-"*80)
            print("SZCZEGÓŁOWA ANALIZA WIADOMOŚCI:")
            print("-"*80)
            
            for i, msg in enumerate(results['messages'], 1):
                print(f"\n[{i}] {msg['speaker']}:")
                print(f"    Tekst: {msg['text'][:100]}{'...' if len(msg['text']) > 100 else ''}")
                print(f"    Sentyment: {msg['sentiment']} (pewność: {msg['confidence']:.2%})")
            
            print("\n" + "-"*80)
        
        print("PODSUMOWANIE DLA KAŻDEGO ROZMÓWCY:")
        print("-"*80)
        
        for speaker, summary in results['speaker_summary'].items():
            stats = summary['statistics']
            print(f"\n{speaker}:")
            print(f"  Dominujący sentyment: {summary['dominant_sentiment']}")
            print(f"  Pozytywne: {stats['positive']} ({stats['positive']/stats['total']*100:.1f}%)")
            print(f"  Negatywne: {stats['negative']} ({stats['negative']/stats['total']*100:.1f}%)")
            print(f"  Neutralne: {stats['neutral']} ({stats['neutral']/stats['total']*100:.1f}%)")
            print(f"  Średnia pewność: {stats['avg_confidence']:.2%}")
        
        print("\n" + "="*80 + "\n")


def main():
    """Główna funkcja programu"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Detektor sentymentu rozmówców w języku polskim'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Ścieżka do pliku z rozmową (opcjonalne)'
    )
    parser.add_argument(
        '--model',
        default='cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual',
        help='Nazwa modelu z Hugging Face (domyślnie: wielojęzyczny RoBERTa)'
    )
    parser.add_argument(
        '--output',
        help='Ścieżka do pliku wyjściowego JSON (opcjonalne)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Wyświetl szczegółową analizę każdej wiadomości (domyślnie: tylko podsumowanie)'
    )
    
    args = parser.parse_args()
    
    # Inicjalizacja detektora
    detector = SentimentDetector(model_name=args.model)
    
    # Analiza rozmowy
    if args.input_file:
        print(f"Analizowanie pliku: {args.input_file}")
        results = detector.analyze_from_file(args.input_file)
    else:
        # Przykładowa rozmowa
        print("Brak pliku wejściowego. Używanie przykładowej rozmowy...")
        example_conversation = """Anna: Cześć! Jak się masz?
Jan: Witaj! Świetnie, dziękuję! A ty?
Anna: Też dobrze! Dziś piękna pogoda, prawda?
Jan: Tak, naprawdę ładny dzień. Planujesz coś na weekend?
Anna: Tak, idę do kina z przyjaciółmi. A ty?
Jan: Niestety muszę pracować, ale to w porządku.
Anna: Szkoda, że nie możesz odpocząć.
Jan: Nie przejmuj się, następnym razem się spotkamy."""
        
        results = detector.analyze_conversation(example_conversation)
    
    # Wyświetlenie wyników
    detector.print_results(results, verbose=args.verbose)
    
    # Zapis do pliku JSON jeśli podano
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wyniki zapisane do: {args.output}")


if __name__ == '__main__':
    main()

