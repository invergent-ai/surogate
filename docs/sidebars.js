module.exports = {
  mySidebar: [
    {
      type: 'doc',
      id: 'index',
      label: 'Welcome',
    },
    {
      type: 'category',
      label: 'About Surogate',
      collapsed: false,
      items: [
        'about/how-it-works',
        'about/adaptive-training',
        'about/dsl',
        'about/automatic-differentiation'
      ],
    },
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/training-modes',
        'getting-started/quickstart-pretraining',
        'getting-started/quickstart-sft',
        'getting-started/quickstart-grpo',
      ],
    },
    {
      type: 'category',
      label: 'Examples Library',
      collapsed: false,
      items: [
        'examples/index',
        {
          type: 'category',
          label: 'Pre-training (PT)',
          items: [
            'examples/pt/qwen3',
          ],
        },
        {
          type: 'category',
          label: 'Fine-tuning (SFT)',
          items: [
            'examples/sft/qwen3-lora',
            'examples/sft/qwen3-qlora',
            'examples/sft/qwen3moe-lora',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'User Guides',
      collapsed: true,
      items: [
        'guides/configuration',
        'guides/datasets',
        'guides/optimizers',
        'guides/precision-and-recipes',
        'guides/qlora',
        'guides/metrics',
        'guides/moe',
        'guides/multi-gpu',
        'guides/multi-node',
        'guides/offloading',
        'guides/memory',
        'guides/debugging',
        'guides/rl-environments',
        'guides/rl-training',
      ],
    },
    {
      type: 'category',
      label: 'Technical Reference',
      collapsed: true,
      items: [
        'reference/benchmarks',
        'reference/python-api',
        'reference/cli',
        'reference/config'
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      collapsed: true,
      items: [
        'appendix/release-notes',
        'appendix/glossary',
        'appendix/compatibility',
        'appendix/faq',
      ],
    },
  ],
};


