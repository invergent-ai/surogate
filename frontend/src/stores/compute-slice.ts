import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { K8Node } from "@/types/compute";
import * as computeApi from "@/api/compute";

export type ComputeSlice = {
    k8sNodes: K8Node[];

    fetchK8Nodes: () => Promise<void>;
}

export const createComputeSlice: StateCreator<AppState, [], [], ComputeSlice> = (set) => ({
    k8sNodes: [] as K8Node[],

    fetchK8Nodes: async () => {
        set({ loading: true, error: null });
        try {
            const res = await computeApi.fetchK8Nodes();
            set({ k8sNodes: res, loading: false });
        } catch (e) {
            set({ error: (e as Error).message, loading: false });
        }
    }
});