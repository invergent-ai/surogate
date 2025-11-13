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
            label: 'Datasets',
            collapsed: false,
            items: [
                {
                    type: 'doc',
                    id: 'datasets/intro',
                    label: 'Introduction'
                },
                {
                    type: 'doc',
                    id: 'datasets/formats',
                    label: 'Dataset Formats'
                }
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
