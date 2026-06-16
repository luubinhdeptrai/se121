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
    # GOM NHÓM: Mỗi dòng (1 mẫu) là 1 Bài Review + Danh sách các ảnh
    df_grouped = df_merged.groupby('review_id').agg({
        'comment_clean': 'first',
        'image_url': list,  # Chuyển thành danh sách các link ảnh
        'overall_satisfaction': 'first',
        'food_score': 'first',
        'service_score': 'first',
        'atmosphere_score': 'first',
        'price_score': 'first'
    }).reset_index()
    
    print(f"Tổng số mẫu (review + list ảnh) sau khi gom nhóm: {len(df_grouped)}")
    
    # Không giới hạn 6000 ở đây nữa. Lấy TOÀN BỘ dữ liệu ban đầu
    df_final = df_grouped.sample(frac=1, random_state=42).reset_index(drop=True)
    sample_size = len(df_final)
    
    # Chia Train (80%), Val (10%), Test (10%) từ TỔNG 6080
    train_size = int(0.8 * sample_size)
    val_size = int(0.1 * sample_size)
    
    train_df, temp_df = train_test_split(df_final, test_size=(sample_size - train_size), random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Lưu kết quả
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Đã chuẩn bị tổng cộng {len(df_final)} mẫu (1 mẫu = 1 text + nhiều ảnh) làm dữ liệu gốc.")
    print(f"- Train: {len(train_df)} mẫu")
    print(f"- Val:   {len(val_df)} mẫu")
    print(f"- Test:  {len(test_df)} mẫu")

if __name__ == '__main__':
    process_data()
