# ============================================================
# OPML RSS LOADER
# ============================================================
import os
import xml.etree.ElementTree as ET


def load_opml(filename):
    """Returns a list of {url, name, group} for every RSS outline in the OPML file."""
    feeds = []
    if not os.path.exists(filename):
        print("[!] Brak OPML:", filename)
        return feeds

    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        body = root.find("body")
        top_level = list(body) if body is not None else []

        for group in top_level:
            group_name = group.get("text") or group.get("title") or "General"
            children = list(group)
            if children:
                for item in children:
                    url = item.get("xmlUrl")
                    name = item.get("title") or item.get("text")
                    if url and name:
                        feeds.append({"url": url, "name": name, "group": group_name})
            else:
                # top-level outline is itself a feed, not a group
                url = group.get("xmlUrl")
                name = group.get("title") or group.get("text")
                if url and name:
                    feeds.append({"url": url, "name": name, "group": "General"})
    except Exception as e:
        print("[!] OPML error:", e)

    return feeds
