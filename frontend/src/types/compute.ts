export interface K8NodeMetrics {
    node_name: string;
    timestamp: number;
    free_memory_bytes?: number;
    total_memory_bytes?: number;
    cpu_utilization_percent?: number;
}

export interface K8Node {
    name: string;
    accelerator_type?: string;
    accelerator_count: number;
    accelerator_available: number;
    cpu_count?: number;
    memory_gb?: number;
    is_ready: boolean;
    metrics?: K8NodeMetrics;
}