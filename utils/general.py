import hashlib

def group_by_file(structured_sources):
    files = {}

    for s in structured_sources:
        if s["file"] not in files:
            files[s["file"]] = s

    return list(files.values())


def get_file_hash(file):
    return hashlib.md5(file.getvalue() if not isinstance(file, bytes) else file).hexdigest()

def make_source_link(file, page):
    # return f"[📄 {file} - pag.{page}](#)"
    return f'<a href="docs/{file}#page={page}" target="_blank">📄 {file} - pag.{page}</a>'

def extract_keywords(query):
    words = re.findall(r"\w+", query.lower())
    return [w for w in words if len(w) > 4]