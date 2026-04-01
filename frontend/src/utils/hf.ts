type HFKind = 'models' | 'datasets';

export interface HFSibling {
  rfilename: string;
}

export interface HFItem {
  id: string;
  author?: string;
  createdAt?: string;
  lastModified?: string;
  last_modified?: string;
  downloads?: number;
  name?: string;
  siblings?: HFSibling[];
  [key: string]: unknown;
}

export async function hfSearch(
    kind: HFKind,
    query: string,
    limit = 10,
    token?: string | null
  ): Promise<HFItem[]> {
    const params = new URLSearchParams();
    params.set('search', query);
    params.set('limit', String(limit));
    // These are commonly supported on the Hub listing endpoints; harmless if ignored:
    params.set('sort', 'downloads');
    params.set('direction', '-1');
    params.set('full', 'true');

    const url = `https://huggingface.co/api/${kind}?${params.toString()}`;

    const headers: Record<string, string> = {};
    if (token) headers['Authorization'] = `Bearer ${token}`;

    const res = await fetch(url, { headers });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`HF ${kind} search failed: ${res.status} ${res.statusText} ${text}`);
    }

    const items = (await res.json()) as HFItem[];
    return items.map((x) => ({
      ...x,
      name: x.id,
      author: x.author ?? x.id?.split('/')?.[0],
      createdAt: x.createdAt ?? x.lastModified ?? x.last_modified,
      downloads: x.downloads ?? 0,
    }));
  }