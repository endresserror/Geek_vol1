import numpy as np
import pickle
from janome.tokenizer import Tokenizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk
import os
import sys

# 和風カラーパレット
WASHI_COLOR = "#F3EFE0"      # 和紙色（背景）
INDIGO_COLOR = "#2A4073"     # 藍色（アクセント）
VERMILION_COLOR = "#E16B56"  # 朱色（警告・失敗）
WAKABA_COLOR = "#A8D39D"     # 若葉色（成功）
INK_COLOR = "#1A1A1A"        # 墨色（テキスト）
BROWN_COLOR = "#6A4028"      # 茶色（サブテキスト）

# 日本語処理用のツール初期化
tokenizer = Tokenizer()
tool = language_tool_python.LanguageTool('ja')

def tokenize(text):
    """テキストを形態素解析して単語リストに分解"""
    tokens = [
        token.base_form
        for token in tokenizer.tokenize(text)
        if token.part_of_speech.split(',')[0] != '記号'
    ]
    return tokens

def preprocess_haiku_file(input_filepath, output_filepath):
    """俳句テキストファイルの前処理とモデル作成"""
    # テキストファイルから俳句を読み込み
    with open(input_filepath, 'r', encoding='utf-8') as f:
        haikus = [line.strip() for line in f if line.strip()]
    
    # 俳句のトークン化
    tokenized_haikus = [tokenize(haiku) for haiku in haikus]
    
    # Word2Vecモデルの学習
    model = Word2Vec(sentences=tokenized_haikus, vector_size=100, window=5, min_count=1, workers=4)
    
    # モデルと俳句をファイルに保存
    with open(output_filepath, 'wb') as f:
        pickle.dump({'model': model, 'haikus': haikus}, f)
    
    print("前処理完了: データを保存しました:", output_filepath)

def evaluate_input_haiku(input_haiku, processed_filepath, threshold=10):
    """入力された俳句の類似度を評価"""
    # 処理済みデータの読み込み
    with open(processed_filepath, 'rb') as f:
        data = pickle.load(f)
    
    model = data['model']
    haikus = data['haikus']
    
    # 入力俳句のトークン化
    input_tokens = tokenize(input_haiku)
    
    # 入力俳句の平均ベクトル計算
    input_vector = np.mean([model.wv[token] for token in input_tokens if token in model.wv], axis=0)
    
    # 既存俳句の平均ベクトル計算
    haiku_vectors = [np.mean([model.wv[token] for token in tokenize(haiku) if token in model.wv], axis=0) for haiku in haikus]
    
    # コサイン類似度の計算
    similarities = cosine_similarity([input_vector], haiku_vectors)
    
    # 平均類似度を計算
    average_similarity = np.mean(similarities)
    similarity_percentage = average_similarity * 100
    
    # 閾値との比較
    is_similar = similarity_percentage >= threshold
    return similarity_percentage, is_similar

def evaluate_meaninglessness(haiku):
    """俳句の意味不明度を評価"""
    # 文法エラーをカウント
    matches = tool.check(haiku)
    error_count = len(matches)
    
    # 単語のつながりの不自然さを評価
    tokens = tokenize(haiku)
    unusual_connections = 0
    
    if len(tokens) >= 3:
        unusual_connections = len(tokens) // 2
    
    # 合計スコアを計算
    meaninglessness_score = error_count + unusual_connections
    
    return meaninglessness_score

def resource_path(relative_path):
    """実行環境に応じたリソースパスを返す"""
    try:
        # PyInstallerでバンドルされている場合
        base_path = sys._MEIPASS
    except Exception:
        # 通常実行時
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

class HaikuEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("俳句認証システム")
        self.root.geometry("800x700")
        self.root.configure(bg=WASHI_COLOR)
        
        # 初期化処理
        self.setup_ui()
        self.processed_file = resource_path('processed_haikus.pkl')
        haiku_file = resource_path('haikus.txt')
        
        # 前処理ファイルの確認と作成
        try:
            with open(self.processed_file, 'rb') as f:
                data = pickle.load(f)
                if 'model' not in data or 'haikus' not in data:
                    raise ValueError("ファイルの形式が正しくありません")
        except (FileNotFoundError, ValueError) as e:
            self.show_status("前処理を実行中です。しばらくお待ちください...")
            preprocess_haiku_file(haiku_file, self.processed_file)
            self.show_status("前処理が完了しました。俳句を入力してください。")
        else:
            self.show_status("認証用の俳句を入力してください。")
    
    def setup_ui(self):
        # フォント設定
        title_font = font.Font(family="MS Gothic", size=22, weight="bold")
        heading_font = font.Font(family="MS Gothic", size=16, weight="bold")
        label_font = font.Font(family="MS Gothic", size=12)
        button_font = font.Font(family="MS Gothic", size=11)
        
        # メインフレーム
        main_frame = tk.Frame(self.root, bg=WASHI_COLOR, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # タイトル部分
        title_frame = tk.Frame(main_frame, bg=INDIGO_COLOR, bd=2, relief=tk.RAISED)
        title_frame.pack(pady=(0, 20), fill=tk.X)
        
        title_label = tk.Label(
            title_frame, 
            text="俳句認証システム", 
            font=title_font, 
            bg=INDIGO_COLOR, 
            fg="white",
            pady=12
        )
        title_label.pack()
        
        # 説明テキスト
        description_frame = tk.Frame(main_frame, bg=WASHI_COLOR, bd=1, relief=tk.GROOVE)
        description_frame.pack(pady=10, fill=tk.X)
        
        description = tk.Label(
            description_frame, 
            text="俳句を入力して認証ボタンをクリックしてください。\n類似度が高いと認証成功となります。",
            font=label_font,
            bg=WASHI_COLOR,
            fg=INK_COLOR,
            pady=10
        )
        description.pack()
        
        # 入力フレーム
        input_frame = tk.Frame(main_frame, bg="#E8DCCA", bd=2, relief=tk.GROOVE)
        input_frame.pack(pady=15, fill=tk.X)
        
        # 入力欄ラベル
        input_label = tk.Label(
            input_frame,
            text="俳句",
            font=heading_font,
            bg="#E8DCCA",
            fg=BROWN_COLOR
        )
        input_label.pack(pady=(10, 5))
        
        # 俳句入力フィールド
        self.haiku_entry = tk.Text(
            input_frame, 
            width=40, 
            height=2,
            font=label_font,
            bg="white",
            fg=INK_COLOR,
            bd=1,
            relief=tk.SUNKEN,
            wrap=tk.WORD
        )
        self.haiku_entry.pack(pady=5, padx=10)
        self.haiku_entry.insert("1.0", "メタ読みだ 泉ピン子の 競走馬")
        
        # 認証ボタン
        button_frame = tk.Frame(input_frame, bg="#E8DCCA")
        button_frame.pack(pady=10)
        
        self.evaluate_button = tk.Button(
            button_frame, 
            text="認証する",
            command=self.evaluate_haiku,
            font=button_font,
            bg=INDIGO_COLOR,
            fg="white",
            padx=20,
            pady=5,
            bd=0,
            relief=tk.RAISED,
            activebackground="#1A2F5A",
            activeforeground="white"
        )
        self.evaluate_button.pack()
        
        # 結果表示フレーム
        self.result_frame = tk.Frame(main_frame, bg=WASHI_COLOR)
        self.result_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # ステータス表示
        status_frame = tk.Frame(main_frame, bg="#749351", bd=1, relief=tk.GROOVE)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(
            status_frame, 
            text="", 
            font=label_font, 
            bg="#749351", 
            fg="white",
            pady=5
        )
        self.status_label.pack(fill=tk.X)
    
    def show_status(self, message):
        """ステータスメッセージを表示"""
        self.status_label.config(text=message)
        self.root.update()
    
    def clear_result_frame(self):
        """結果表示領域をクリア"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()
    
    def evaluate_haiku(self):
        """俳句の評価と結果表示"""
        self.clear_result_frame()
        input_haiku = self.haiku_entry.get("1.0", "end-1c")
        
        if not input_haiku.strip():
            messagebox.showerror("エラー", "俳句を入力してください。")
            return
        
        self.show_status("認証中...")
        
        try:
            similarity_percentage, is_similar = evaluate_input_haiku(input_haiku, self.processed_file)
            tokens = tokenize(input_haiku)
            meaninglessness_score = evaluate_meaninglessness(input_haiku)
            
            # 結果表示の準備
            result_font = font.Font(family="MS Gothic", size=12)
            result_heading = font.Font(family="MS Gothic", size=14, weight="bold")
            
            # 結果表示フレーム
            result_container = tk.Frame(self.result_frame, bg=WASHI_COLOR)
            result_container.pack(pady=10, fill=tk.BOTH, expand=True)
            
            # 評価結果のヘッダー
            result_header = tk.Label(
                result_container,
                text="認証結果",
                font=result_heading,
                bg=INDIGO_COLOR,
                fg="white",
                pady=8
            )
            result_header.pack(fill=tk.X)
            
            if is_similar:  # 認証成功時
                # 認証成功画面
                success_frame = tk.Frame(result_container, bg=WAKABA_COLOR, bd=1, relief=tk.GROOVE)
                success_frame.pack(fill=tk.X, padx=10, pady=10)
                
                success_label = tk.Label(
                    success_frame,
                    text="認証成功！あなたの書いた俳句として認められました。",
                    font=font.Font(family="MS Gothic", size=14, weight="bold"),
                    fg=BROWN_COLOR,
                    bg=WAKABA_COLOR,
                    pady=10
                )
                success_label.pack()
                
                # 分解語と意味不明度のみ表示
                info_frame = tk.Frame(success_frame, bg=WAKABA_COLOR)
                info_frame.pack(pady=5)
                
                info_text = f"分解語: {', '.join(tokens)}\n"
                info_text += f"意味不明度: {meaninglessness_score}"
                
                info_label = tk.Label(
                    info_frame,
                    text=info_text,
                    font=result_font,
                    fg=BROWN_COLOR,
                    bg=WAKABA_COLOR,
                    justify=tk.LEFT
                )
                info_label.pack(pady=5)
                
                # 評価コメント
                if meaninglessness_score >= 5:
                    comment = "高い意味不明度も持つ優れた俳句です！"
                else:
                    comment = "入力お疲れさまでした"
                    
                comment_label = tk.Label(
                    success_frame,
                    text=comment,
                    font=result_font,
                    fg=BROWN_COLOR,
                    bg=WAKABA_COLOR
                )
                comment_label.pack(pady=(0, 10))
                
                # 装飾線
                separator = tk.Frame(success_frame, height=2, bg=BROWN_COLOR)
                separator.pack(fill=tk.X, padx=50, pady=5)
                
                new_button = tk.Button(
                    success_frame,
                    text="新しい認証を行う",
                    command=self.reset_input,
                    font=result_font,
                    bg="#477A47",
                    fg="white",
                    padx=15,
                    pady=5,
                    bd=0,
                    relief=tk.FLAT,
                    activebackground="#2F522F",
                    activeforeground="white"
                )
                new_button.pack(pady=10)
                
                self.show_status("認証完了: 認証に成功しました！")
            else:  # 認証失敗時 - 変更部分
                # 認証失敗画面 - 詳細フレームは表示せず直接失敗画面を表示
                failure_frame = tk.Frame(result_container, bg="#F9E9E7", bd=1, relief=tk.GROOVE)
                failure_frame.pack(fill=tk.X, padx=10, pady=10)
                
                failure_label = tk.Label(
                    failure_frame,
                    text="認証失敗！基準を満たしていません。",  # 類似度の数値に言及しない
                    font=font.Font(family="MS Gothic", size=14, weight="bold"),
                    fg=VERMILION_COLOR,
                    bg="#F9E9E7",
                    pady=10
                )
                failure_label.pack()
                
                # 分解語と意味不明度のみ表示
                info_frame = tk.Frame(failure_frame, bg="#F9E9E7")
                info_frame.pack(pady=5)
                
                info_text = f"分解語: {', '.join(tokens)}\n"
                info_text += f"意味不明度: {meaninglessness_score}"
                
                info_label = tk.Label(
                    info_frame,
                    text=info_text,
                    font=result_font,
                    fg=VERMILION_COLOR,
                    bg="#F9E9E7",
                    justify=tk.LEFT
                )
                info_label.pack(pady=5)
                
                # 改善アドバイス
                advice_text = "データベースの俳句に近い表現を試してみてください。\n別の俳句で再度認証を行ってください。"
                advice_label = tk.Label(
                    failure_frame,
                    text=advice_text,
                    font=result_font,
                    fg=VERMILION_COLOR,
                    bg="#F9E9E7",
                    wraplength=450
                )
                advice_label.pack(pady=(0, 10))
                
                # 装飾線
                separator = tk.Frame(failure_frame, height=2, bg=VERMILION_COLOR)
                separator.pack(fill=tk.X, padx=50, pady=5)
                
                retry_button = tk.Button(
                    failure_frame,
                    text="再認証する",
                    command=self.reset_input,
                    font=result_font,
                    bg=VERMILION_COLOR,
                    fg="white",
                    padx=15,
                    pady=5,
                    bd=0,
                    relief=tk.FLAT,
                    activebackground="#B05242",
                    activeforeground="white"
                )
                retry_button.pack(pady=10)
                
                self.show_status("認証完了: 認証に失敗しました。別の俳句で再試行してください。")
                
        except Exception as e:
            # エラー処理
            error_frame = tk.Frame(self.result_frame, bg="#F9E9E7", bd=1, relief=tk.GROOVE)
            error_frame.pack(fill=tk.X, padx=10, pady=10)
            
            error_label = tk.Label(
                error_frame,
                text=f"エラーが発生しました: {str(e)}\nhaikus.txtファイルがあるか確認してください。",
                font=font.Font(family="MS Gothic", size=12),
                fg=VERMILION_COLOR,
                bg="#F9E9E7",
                pady=10,
                wraplength=500
            )
            error_label.pack()
            
            self.show_status("エラーが発生しました。詳細を確認してください。")
    
    def reset_input(self):
        """入力と結果をリセット"""
        self.haiku_entry.delete("1.0", tk.END)
        self.clear_result_frame()
        self.show_status("新しい俳句を入力してください。")


if __name__ == '__main__':
    root = tk.Tk()
    app = HaikuEvaluatorApp(root)
    root.mainloop()
