import os
import pickle
import mimetypes
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request
import io

# ========================
# CONFIG
# ========================
CLIENT_SECRET_FILE = r"extract/driver token/drive_pdd.json"
TOKEN_PATH = r"extract/driver token/token.pickle"

SCOPES = ['https://www.googleapis.com/auth/drive']

# ========================
# AUTH
# ========================

def get_drive_service():
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print('dcm ')
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)

        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)



# ========================
# UPLOAD FILE
# ========================

def upload_file(file_path, folder_id=None):
    drive_service = get_drive_service()

    file_name = os.path.basename(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)

    file_metadata = {
        'name': file_name
    }

    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    print("✅ Uploaded:", file_name)
    print("📁 File ID:", file.get('id'))
    return file.get('id')

# ========================
# DOWNLOAD FILE
# ========================
import io
from googleapiclient.http import MediaIoBaseDownload

def download_video(file_id, save_path):
    drive_service = get_drive_service()

    # CHỈNH SỬA DÒNG NÀY: Dùng get với alt='media'
    request = drive_service.files().get(fileId=file_id, alt='media')
    
    with io.FileIO(save_path, 'wb') as fh:
        # Tăng chunk_size lên 20MB cho bốc
        downloader = MediaIoBaseDownload(fh, request, chunksize=20*1024*1024)
        
        done = False
        print(f"Bắt đầu tải video: {save_path}")
        
        try:
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    # Sửa lại cách in progress để chính xác hơn
                    print(f"Đã tải: {int(status.progress() * 100)}%")
            print("✅ Tải hoàn tất!")
        except Exception as e:
            print(f"❌ Lỗi khi tải: {e}")

def download_file(file_id, save_path):
    drive_service = get_drive_service()
    
    request = drive_service.files().get_media(fileId=file_id)
    file_io = io.FileIO(save_path, 'wb')
    downloader = MediaIoBaseDownload(file_io, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading: {int(status.progress() * 100)}%")

    print("✅ Download completed:", save_path)

# ========================
# DELETE FILE
# ========================

def delete_file(file_id):
    drive_service = get_drive_service()

    drive_service.files().delete(fileId=file_id).execute()
    print("🗑️ File deleted:", file_id)

# ========================
# LIST FILES
# ========================

def list_files():
    drive_service = get_drive_service()

    results = drive_service.files().list(
        pageSize=20,
        fields="files(id, name, mimeType)"
    ).execute()

    items = results.get('files', [])

    if not items:
        print("No files found.")
        return

    for item in items:
        print(f"{item['name']} | {item['mimeType']} | {item['id']}")

# ========================
# RENAME FILE
# ========================

def rename_file(file_id, new_name):
    drive_service = get_drive_service()

    file = drive_service.files().update(
        fileId=file_id,
        body={'name': new_name}
    ).execute()

    print("✏️ Renamed to:", new_name)
def list_all_files_with_id(folder_id=None):
    drive_service = get_drive_service()

    file_data = {} # Dùng dictionary: { "tên_file": "id_file" }
    page_token = None
    
    query = f"'{folder_id}' in parents and trashed = false" if folder_id else "trashed = false"

    print(f"--- Đang quét danh sách file và ID... ---")
    
    while True:
        results = drive_service.files().list(
            q=query,
            pageSize=1000,
            # THAY ĐỔI: Thêm 'id' vào fields
            fields="nextPageToken, files(id, name)", 
            pageToken=page_token
        ).execute()

        items = results.get('files', [])
        for item in items:
            # Lưu tên file làm key, ID làm value
            file_data[item['name']] = item['id']

        page_token = results.get('nextPageToken', None)
        if not page_token:
            break
            
    print(f"✅ Tổng cộng: {len(file_data)} files.")
    return file_data

# ========================
# CÁCH DÙNG
# ========================
# data = list_all_files_with_id('FOLDER_ID_CUA_BAN')

# 1. Xem ID của một file cụ thể
# print(data.get('v_G_rIn8K73ls.mp4')) 

# 2. Lấy danh sách tên file (giống os.listdir)
# filenames = list(data.keys())

# ========================
# CÁCH SỬ DỤNG
# ========================
# Giả sử bạn muốn lấy hết tên video trong folder ActivityNet
# folder_id = '1LGldjbC7x6NLgevaTF7tuSfYbSWxneYv' # Thay bằng ID folder của bạn
# all_videos = list_all_filenames(folder_id)

# In thử 5 file đầu tiên
# print(all_videos[:5])

# fileid = upload_file(r'C:\PDD\python\test.mp4')
# download_file('1LGldjbC7x6NLgevaTF7tuSfYbSWxneYv', 'test.mp4')


# delete_file('1IEH0tZNfhh1LQ5sy1rq58VnVTknmOkxt')
# all_videos = list_all_files_with_id('14yuk3BTCVgqsWJPSpaxMDu2Lmv7LpjjS')
# print(len(all_videos))
# print(list(all_videos.keys())[:5], list(all_videos.values())[:5])
# download_file('1FM2mXksAnUYrCEt0Lj34GQecabVTgvUM', 'test2.pt')