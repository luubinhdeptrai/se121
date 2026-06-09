import pandas as pd
import os

def process_data():
    checkpoints_dir = '/Users/abc/Documents/SE/checkpoints_clean'
    output_dir = '/Users/abc/Documents/SE/Multimodal-Sentiment-Analysis/data'
    
    # Đọc dữ liệu
    df_multi = pd.read_csv(os.path.join(checkpoints_dir, 'multimodal_reviews.csv'))
    df_reviews = pd.read_csv(os.path.join(checkpoints_dir, 'reviews_clean.csv'))
    
    print(f"Đọc {len(df_multi)} dòng từ multimodal_reviews.csv")
    print(f"Đọc {len(df_reviews)} dòng từ reviews_clean.csv")
    
    # Merge để lấy các điểm số (food, price, atmosphere) từ reviews_clean vào df_multi
    df_merged = pd.merge(
        df_multi, 
        df_reviews[['review_id', 'food_score', 'price_score', 'atmosphere_score']], 
        on='review_id', 
        how='inner'
    )
    
    # Lọc bỏ các cột bị missing values ở những trường quan trọng
    cols_to_check = ['comment_clean', 'image_url', 'avg_rating', 'food_score', 'price_score', 'atmosphere_score']
    df_merged = df_merged.dropna(subset=cols_to_check)
    
    # Chỉ giữ lại các cột cần thiết cho việc train
    df_final = df_merged[['review_id', 'comment_clean', 'image_url', 'avg_rating', 'food_score', 'price_score', 'atmosphere_score']].copy()
    
    # Reset index và lấy 5000 mẫu ngẫu nhiên
    df_final = df_final.sample(n=5000, random_state=42).reset_index(drop=True)
    
    output_path = os.path.join(output_dir, 'processed_reviews.csv')
    df_final.to_csv(output_path, index=False)
    print(f"Đã xử lý xong. Dữ liệu cuối cùng có {len(df_final)} dòng.")
    print(f"Lưu tại: {output_path}")

if __name__ == '__main__':
    process_data()
