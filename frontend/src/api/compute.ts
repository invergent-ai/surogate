import type { K8Node } from "@/types/compute";
import { authFetch } from "@/api/auth";

export async function fetchK8Nodes(): Promise<K8Node[]> {
    const response = await authFetch(`/api/compute/nodes`);
    return (await response.json()) as K8Node[];
}