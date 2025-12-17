#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Podsumowywacz rozmów w języku polskim
Używa modelu AI do automatycznego podsumowania rozmów z plików tekstowych
"""

import re
import json
import time
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


class ConversationSummarizer:
    """Klasa do podsumowywania rozmów po polsku"""
    
    def __init__(self, model_name: str = "facebook/mbart-large-50-many-to-many-mmt"):
        """
        Inicjalizacja podsumowywacza rozmów
        
        Args:
            model_name: Nazwa modelu z Hugging Face. Domyślnie używa wielojęzycznego
                       modelu mBART, który dobrze działa z polskim.
                       
                       Rekomendowane modele do podsumowywania po polsku:
                       - 'facebook/mbart-large-50-many-to-many-mmt' (domyślny)
                         - Wielojęzyczny mBART, wspiera polski (pl_XX)
                       - 'google/mt5-base'
                         - Wielojęzyczny T5, wspiera polski
                       - 'google/mt5-small'
                         - Lżejsza wersja MT5
        """
        print(f"Ładowanie modelu: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używane urządzenie: {self.device}")
        
        # Tworzenie pipeline do podsumowywania
        self.summarizer = pipeline(
            "summarization",
            model=model_name,
            tokenizer=model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        self.model_name = model_name
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
    
    def _prepare_text_for_summarization(self, text: str, max_length: int = 1024) -> str:
        """
        Przygotowuje tekst do podsumowania, obcinając go jeśli jest za długi
        
        Args:
            text: Tekst do przygotowania
            max_length: Maksymalna długość tekstu w znakach
        
        Returns:
            Przygotowany tekst
        """
        if len(text) > max_length:
            # Obcinamy do max_length, ale staramy się obciąć na końcu zdania
            truncated = text[:max_length]
            last_period = truncated.rfind('.')
            last_exclamation = truncated.rfind('!')
            last_question = truncated.rfind('?')
            
            last_sentence_end = max(last_period, last_exclamation, last_question)
            if last_sentence_end > max_length * 0.8:  # Jeśli znaleźliśmy koniec zdania w ostatnich 20%
                return truncated[:last_sentence_end + 1]
            return truncated
        return text
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 30) -> Dict[str, any]:
        """
        Podsumowuje pojedynczy tekst
        
        Args:
            text: Tekst do podsumowania
            max_length: Maksymalna długość podsumowania w tokenach
            min_length: Minimalna długość podsumowania w tokenach
        
        Returns:
            Słownik z wynikami podsumowania
        """
        if not text or not text.strip():
            return {
                'summary': '',
                'original_length': 0,
                'summary_length': 0,
                'error': 'Pusty tekst'
            }
        
        # Przygotowanie tekstu
        prepared_text = self._prepare_text_for_summarization(text, max_length=1024)
        
        try:
            # Używamy pipeline do podsumowania
            # Pipeline automatycznie obsługuje różne modele (mBART, MT5, itp.)
            result = self.summarizer(
                prepared_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary_text = result[0]['summary_text'] if isinstance(result, list) else result['summary_text']
            
            return {
                'summary': summary_text,
                'original_length': len(text),
                'summary_length': len(summary_text),
                'compression_ratio': len(summary_text) / len(text) if len(text) > 0 else 0
            }
        except Exception as e:
            print(f"Błąd podczas podsumowywania: {e}")
            # Dla modeli mBART może być potrzebne specjalne formatowanie
            if 'mbart' in self.model_name.lower():
                try:
                    # Próba z bezpośrednim użyciem tokenizera i modelu
                    tokenizer = self.summarizer.tokenizer
                    model = self.summarizer.model
                    
                    # Ustawienie języka źródłowego na polski
                    if hasattr(tokenizer, 'src_lang'):
                        tokenizer.src_lang = "pl_XX"
                    
                    # Tokenizacja
                    inputs = tokenizer(prepared_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    # Generowanie z kodem języka
                    generate_kwargs = {
                        'max_length': max_length,
                        'min_length': min_length,
                        'do_sample': False,
                        'num_beams': 4
                    }
                    
                    if hasattr(tokenizer, 'lang_code_to_id') and 'pl_XX' in tokenizer.lang_code_to_id:
                        generate_kwargs['forced_bos_token_id'] = tokenizer.lang_code_to_id['pl_XX']
                    
                    generated_tokens = model.generate(**inputs, **generate_kwargs)
                    summary_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
                    
                    return {
                        'summary': summary_text,
                        'original_length': len(text),
                        'summary_length': len(summary_text),
                        'compression_ratio': len(summary_text) / len(text) if len(text) > 0 else 0
                    }
                except Exception as e2:
                    print(f"Błąd podczas specjalnego formatowania mBART: {e2}")
            
            return {
                'summary': '',
                'original_length': len(text),
                'summary_length': 0,
                'error': str(e)
            }
    
    def summarize_conversation(self, conversation_text: str, 
                              max_length: int = 200, 
                              min_length: int = 50,
                              include_speakers: bool = True) -> Dict[str, any]:
        """
        Podsumowuje całą rozmowę
        
        Args:
            conversation_text: Tekst rozmowy
            max_length: Maksymalna długość podsumowania w tokenach
            min_length: Minimalna długość podsumowania w tokenach
            include_speakers: Czy uwzględniać informacje o rozmówcach w podsumowaniu
        
        Returns:
            Słownik z wynikami podsumowania
        """
        messages = self.parse_conversation(conversation_text)
        
        if not messages:
            return {
                'error': 'Brak wiadomości w rozmowie',
                'summary': '',
                'messages': []
            }
        
        # Zbieranie informacji o rozmówcach
        speakers = list(set([msg['speaker'] for msg in messages]))
        total_messages = len(messages)
        
        # Tworzenie pełnego tekstu rozmowy do podsumowania
        if include_speakers:
            # Format z nazwami rozmówców
            full_text = '\n'.join([f"{msg['speaker']}: {msg['text']}" for msg in messages])
        else:
            # Tylko teksty bez nazw
            full_text = '\n'.join([msg['text'] for msg in messages])
        
        # Rozpoczęcie pomiaru czasu
        start_time = time.time()
        
        # Podsumowanie
        summary_result = self.summarize_text(full_text, max_length=max_length, min_length=min_length)
        
        # Zakończenie pomiaru czasu
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Statystyki rozmowy
        total_chars = sum(len(msg['text']) for msg in messages)
        avg_message_length = total_chars / total_messages if total_messages > 0 else 0
        
        return {
            'summary': summary_result.get('summary', ''),
            'original_length': summary_result.get('original_length', 0),
            'summary_length': summary_result.get('summary_length', 0),
            'compression_ratio': summary_result.get('compression_ratio', 0),
            'speakers': speakers,
            'total_messages': total_messages,
            'total_characters': total_chars,
            'average_message_length': avg_message_length,
            'processing_time_seconds': processing_time,
            'messages': messages[:10] if len(messages) > 10 else messages,  # Pierwsze 10 wiadomości jako przykłady
            'error': summary_result.get('error')
        }
    
    def summarize_from_file(self, file_path: str, 
                           max_length: int = 200, 
                           min_length: int = 50,
                           include_speakers: bool = True) -> Dict[str, any]:
        """
        Podsumowuje rozmowę z pliku tekstowego
        
        Args:
            file_path: Ścieżka do pliku z rozmową
            max_length: Maksymalna długość podsumowania w tokenach
            min_length: Minimalna długość podsumowania w tokenach
            include_speakers: Czy uwzględniać informacje o rozmówcach
        
        Returns:
            Wyniki podsumowania
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation_text = f.read()
            return self.summarize_conversation(
                conversation_text, 
                max_length=max_length,
                min_length=min_length,
                include_speakers=include_speakers
            )
        except Exception as e:
            return {
                'error': f'Błąd podczas czytania pliku: {str(e)}',
                'summary': ''
            }
    
    def print_results(self, results: Dict[str, any]):
        """
        Wyświetla wyniki podsumowania w czytelnej formie
        
        Args:
            results: Wyniki z metody summarize_conversation
        """
        if 'error' in results and results['error']:
            print(f"Błąd: {results['error']}")
            return
        
        print("\n" + "="*80)
        print("PODSUMOWANIE ROZMOWY")
        print("="*80)
        
        print(f"\nLiczba rozmówców: {len(results.get('speakers', []))}")
        print(f"Rozmówcy: {', '.join(results.get('speakers', []))}")
        print(f"Łączna liczba wiadomości: {results.get('total_messages', 0)}")
        print(f"Łączna liczba znaków: {results.get('total_characters', 0):,}")
        print(f"Średnia długość wiadomości: {results.get('average_message_length', 0):.1f} znaków")
        
        if 'processing_time_seconds' in results:
            processing_time = results['processing_time_seconds']
            if processing_time < 1:
                print(f"Czas przetwarzania: {processing_time * 1000:.2f} ms")
            else:
                print(f"Czas przetwarzania: {processing_time:.2f} s")
        
        if 'compression_ratio' in results:
            ratio = results['compression_ratio']
            print(f"Współczynnik kompresji: {ratio:.2%}")
        
        print("\n" + "-"*80)
        print("PODSUMOWANIE:")
        print("-"*80)
        print(results.get('summary', 'Brak podsumowania'))
        print("-"*80)
        
        print("\n" + "="*80 + "\n")


def main():
    """Główna funkcja programu"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Podsumowywacz rozmów w języku polskim'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='Ścieżka do pliku z rozmową (opcjonalne)'
    )
    parser.add_argument(
        '--model',
        default='facebook/mbart-large-50-many-to-many-mmt',
        help='Nazwa modelu z Hugging Face do podsumowywania. '
             'Domyślnie: facebook/mbart-large-50-many-to-many-mmt. '
             'Alternatywy: google/mt5-base, google/mt5-small'
    )
    parser.add_argument(
        '--output',
        help='Ścieżka do pliku wyjściowego JSON (opcjonalne)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=200,
        help='Maksymalna długość podsumowania w tokenach (domyślnie: 200)'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimalna długość podsumowania w tokenach (domyślnie: 50)'
    )
    parser.add_argument(
        '--no-speakers',
        action='store_true',
        help='Nie uwzględniaj nazw rozmówców w podsumowaniu'
    )
    
    args = parser.parse_args()
    
    # Inicjalizacja podsumowywacza
    summarizer = ConversationSummarizer(model_name=args.model)
    
    # Podsumowanie rozmowy
    if args.input_file:
        print(f"Podsumowywanie pliku: {args.input_file}")
        results = summarizer.summarize_from_file(
            args.input_file,
            max_length=args.max_length,
            min_length=args.min_length,
            include_speakers=not args.no_speakers
        )
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
        
        results = summarizer.summarize_conversation(
            example_conversation,
            max_length=args.max_length,
            min_length=args.min_length,
            include_speakers=not args.no_speakers
        )
    
    # Wyświetlenie wyników
    summarizer.print_results(results)
    
    # Zapis do pliku JSON jeśli podano
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wyniki zapisane do: {args.output}")


if __name__ == '__main__':
    main()
