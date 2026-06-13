import pandas as pd
import os
from sklearn.model_selection import train_test_split

def process_data():
    raw_dir = './data_raw'
    output_dir = './data/text'
    os.makedirs(output_dir, exist_ok=True)
    
    # Đọc dữ liệu
    df_multi = pd.read_csv(os.path.join(raw_dir, 'multimodal_reviews.csv'))
    df_reviews = pd.read_csv('./data_processed/reviews_clean_enhanced.csv')
    
    print(f"Đọc {len(df_multi)} dòng từ multimodal_reviews.csv")
    print(f"Đọc {len(df_reviews)} dòng từ reviews_clean_enhanced.csv")
    
    # Merge để lấy các điểm số từ reviews_clean_enhanced vào df_multi
    cols_to_merge = ['review_id', 'food_score', 'service_score', 'atmosphere_score', 'price_score', 'overall_satisfaction']
    
    df_merged = pd.merge(
        df_multi, 
        df_reviews[cols_to_merge], 
        on='review_id', 
        how='inner'
    )
    
    # Lọc bỏ các cột bị missing values ở những trường quan trọng
    cols_to_check = ['comment_clean', 'image_url', 'overall_satisfaction', 'food_score', 'service_score', 'atmosphere_score', 'price_score']
    df_merged = df_merged.dropna(subset=cols_to_check)
    
    # ĐẢM BẢO KHÔNG CÓ LINK ẢNH TRÙNG LẶP: Mỗi dòng phải là 1 ảnh độc nhất
    df_merged = df_merged.drop_duplicates(subset=['image_url'])
    
    # Chỉ giữ lại các cột cần thiết cho việc train
    final_cols = ['review_id', 'comment_clean', 'image_url', 'overall_satisfaction', 'food_score', 'service_score', 'atmosphere_score', 'price_score']
    df_final = df_merged[final_cols].copy()
    
    # Lấy 5500 mẫu ngẫu nhiên (dư ra 500 mẫu làm backup cho các link ảnh chết)
    df_final = df_final.sample(n=5500, random_state=42).reset_index(drop=True)
    
    # Chia Train, Val, Test
    train_df, temp_df = train_test_split(df_final, test_size=1100, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=550, random_state=42)
    
    # Lưu kết quả
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Đã chuẩn bị {len(df_final)} dòng (bao gồm dự phòng).")
    print(f"- Train: {len(train_df)} dòng")
    print(f"- Val:   {len(val_df)} dòng")
    print(f"- Test:  {len(test_df)} dòng")

if __name__ == '__main__':
    process_data()
