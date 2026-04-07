type HFKind = 'models' | 'datasets';

export const GGUF_QUANTS = {
  '1bit': ['UD-IQ1_S', 'UD-IQ1_M', 'IQ1_M'],
  '2bit': ['UD-UD-IQ2_XXS', 'Q2_K', 'UD-IQ2_M', 'Q2_K_L', 'UD-Q2_K_XL', 'UD-IQ2_XXS', 'IQ2_M'],
  '3bit': ['UD-IQ3_XXS', 'Q3_K_S', 'Q3_K_M', 'Q3_K_L', 'UD-Q3_K_XL', 'UD-IQ3_S', 'IQ3_XXS', 'IQ3_M'],
  '4bit': ['IQ4_XS', 'Q4_K_S', 'IQ4_NL', 'Q4_0', 'Q4_1', 'Q4_K_M', 'UD-Q4_K_XL', 'UD-Q4_K_L', 'UD-IQ4_NL', 'UD-IQ4_XS'],
  '5bit': ['Q5_K_S', 'Q5_K_M', 'UD-Q5_K_XL'],
  '6bit': ['Q6_K', 'Q6_K_XL', 'UD-Q6_K_XL', 'UD-Q6_K_S'],
  '8bit': ['Q8_0', 'UD-Q8_K_XL'],
} as const;

export const GGUF_PREFERRED_QUANTS = [
  // UD variants (best quality per bit) -- Q4 is the sweet spot
  "UD-Q4_K_XL", "UD-Q4_K_L", "UD-Q5_K_XL", "UD-Q3_K_XL", "UD-Q6_K_XL", "UD-Q6_K_S", "UD-Q8_K_XL", "UD-Q2_K_XL", 
  "UD-IQ4_NL", "UD-IQ4_XS", "UD-IQ3_S", "UD-IQ3_XXS", "UD-IQ2_M", "UD-IQ2_XXS", "UD-IQ1_M", "UD-IQ1_S",
  // Standard quants (fallback for non-Unsloth repos)
  "Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q3_K_M", "Q3_K_L", "Q3_K_S", "Q2_K", "Q2_K_L",
  "IQ4_NL", "IQ4_XS", "IQ3_M", "IQ3_XXS", "IQ2_M", "IQ1_M"
] as const;

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