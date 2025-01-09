import gdown

file_id = "1GJv_gqF2ptjGYM5DzAjTZqJBVNbYs5Zf"
url = f"https://drive.google.com/uc?id={file_id}"

gdown.download(url, quiet=False)

