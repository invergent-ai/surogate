module.exports = {
    mySidebar: [
        {
            type: 'category',
            label: 'Getting Started',
            collapsed: false,
            items: [
                {
                    type: 'doc',
                    id: 'getting-started/overview',
                    label: 'Overview'
                },
                {
                    type: 'doc',
                    id: 'getting-started/configuration',
                    label: 'Configuration'
                },
            ]
        },
        {
            type: 'category',
            label: 'VLM/LLM Evaluation',
            collapsed: false,
            items: [
                {
                    type: 'doc',
                    id: 'eval/overview',
                    label: 'Overview'
                },
                {
                    type: 'doc',
                    id: 'eval/model-support',
                    label: 'Model Support'
                },
                {
                    type: 'doc',
                    id: 'eval/datasets',
                    label: 'Datasets'
                },
                {
                    type: 'doc',
                    id: 'eval/functional-metrics',
                    label: 'Functional Metrics'
                },
                {
                    type: 'doc',
                    id: 'eval/performance-testing',
                    label: 'Performance Testing'
                },
                {
                    type: 'doc',
                    id: 'eval/standard-benchmarks',
                    label: 'Standard Benchmarks'
                },
                {
                    type: 'doc',
                    id: 'eval/third-party-benchmarks',
                    label: 'Third-Party Benchmarks'
                },
                {
                    type: 'doc',
                    id: 'eval/red-teaming',
                    label: 'Red Teaming & Security'
                },
                {
                    type: 'doc',
                    id: 'eval/guardrails',
                    label: 'Guardrails'
                },
                {
                    type: 'doc',
                    id: 'eval/crafting-configs',
                    label: 'Crafting Configs'
                },
                {
                    type: 'doc',
                    id: 'eval/results',
                    label: 'Results & Reporting'
                },
            ]
        },
        {
            type: 'category',
            label: 'Quantization',
            items: [
                {
                    type: 'doc',
                    id: 'ptq/intro',
                    label: 'Introduction'
                }
            ]
        },
        {
            type: 'category',
            label: 'Support',
            items: [
                {
                    type: 'doc',
                    label: 'Connect with us',
                    id: 'support/connect',
                },
            ]
        }
    ],
};