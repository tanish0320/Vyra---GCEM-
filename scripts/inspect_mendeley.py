import urllib.request
import re
import json

def check_mendeley(doi_or_id):
    url = f"https://data.mendeley.com/datasets/{doi_or_id}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            print(f"[{doi_or_id}] Status: {resp.status}, HTML length: {len(html)}")
            
            # Check for direct file download links
            # Mendeley Data public file links usually look like:
            # https://data.mendeley.com/public-files/datasets/{id}/files/{file_id}/file_downloaded
            file_ids = re.findall(r'/public-files/datasets/[^/]+/files/([^/]+)/file_downloaded', html)
            print(f"[{doi_or_id}] Found file_downloaded links: {file_ids}")
            
            # Check for embedded JSON
            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">([^<]+)</script>', html)
            if match:
                data = json.loads(match.group(1))
                page_props = data.get('props', {}).get('pageProps', {})
                dataset_data = page_props.get('dataset', {})
                files = dataset_data.get('files', [])
                print(f"[{doi_or_id}] NEXT_DATA files: {len(files)}")
                for f in files:
                    print("  File:", f.get('filename') or f.get('name'), "Size:", f.get('size'), "Download URL:", f.get('downloadUrl') or f.get('download_url'))
            else:
                print(f"[{doi_or_id}] __NEXT_DATA__ not found")
    except Exception as e:
        print(f"[{doi_or_id}] Error: {e}")

if __name__ == '__main__':
    check_mendeley('m53vz6b7fx/2')
    check_mendeley('2xx4j3kjg2/1')
