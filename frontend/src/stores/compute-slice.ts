import type { StateCreator } from "zustand";
import type { AppState } from "./app-store";
import type { K8Node } from "@/types/compute";
import type { CloudBackend, InstanceOffer } from "@/api/compute";
import * as computeApi from "@/api/compute";

export type ComputeSlice = {
    k8sNodes: K8Node[];
    cloudBackends: CloudBackend[];
    backendOffers: InstanceOffer[];
    backendOffersLoading: boolean;
    backendOffersError: string | null;

    fetchK8Nodes: () => Promise<void>;
    fetchCloudBackends: () => Promise<void>;
    fetchBackendOffers: () => Promise<void>;
    setBackendOffers: (offers: InstanceOffer[]) => void;
    deleteCloudBackend: (backendType: string) => Promise<void>;
}

export const createComputeSlice: StateCreator<AppState, [], [], ComputeSlice> = (set, get) => ({
    k8sNodes: [] as K8Node[],
    cloudBackends: [] as CloudBackend[],
    backendOffers: [] as InstanceOffer[],
    backendOffersLoading: false,
    backendOffersError: null as string | null,

    fetchK8Nodes: async () => {
        try {
            const res = await computeApi.fetchK8Nodes();
            set({ k8sNodes: res });
        } catch (e) {
            set({ error: (e as Error).message });
        }
    },

    fetchCloudBackends: async () => {
        const projectId = get().activeProjectId;
        if (!projectId) return;
        try {
            const res = await computeApi.fetchCloudBackends(projectId);
            set({ cloudBackends: res });
        } catch (e) {
            set({ error: (e as Error).message });
        }
    },

    setBackendOffers: (offers) => set({ backendOffers: offers, backendOffersError: null }),

    deleteCloudBackend: async (backendType) => {
        const projectId = get().activeProjectId;
        if (!projectId) return;
        await computeApi.deleteCloudBackend(projectId, backendType);
        set((s) => ({ cloudBackends: s.cloudBackends.filter((b) => b.type !== backendType) }));
    },

    fetchBackendOffers: async () => {
        const projectId = get().activeProjectId;
        if (!projectId) return;
        set({ backendOffersLoading: true, backendOffersError: null });
        try {
            const res = await computeApi.fetchBackendOffers(projectId);
            set({ backendOffers: res });
        } catch (e) {
            set({ backendOffersError: (e as Error).message });
        } finally {
            set({ backendOffersLoading: false });
        }
    },
});
