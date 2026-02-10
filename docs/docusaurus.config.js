import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';


/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
    title: 'Surogate',
    tagline: 'Insanely fast LLM pre-training and fine-tuning for modern NVIDIA GPUs',
    url: 'https://docs.surogate.ai',
    baseUrl: '/',
    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'throw',
    favicon: 'img/favicon.ico',
    organizationName: 'surogate', // Usually your GitHub org/user name.
    projectName: 'surogate-docs', // Usually your repo name.
    deploymentBranch: 'gh-pages',
    trailingSlash: false,
    markdown: {
        mermaid: true,
    },
    themes: [
        '@docusaurus/theme-mermaid',
    ],
    i18n: {
        defaultLocale: "en",
        locales: ["en"],
        localeConfigs: {
            en: {
                label: "English",
            }
        },
    },
    themeConfig: {
        colorMode: {
            // "light" | "dark"
            defaultMode: "dark",
            // Use user preference instead of default mode
            respectPrefersColorScheme: true,
            // Hides the switch in the navbar
            // Useful if you want to support a single color mode
            disableSwitch: false,
        },
        navbar: {
            title: "Documentation",
            logo: {
                alt: 'logo',
                src: 'img/logo-black.svg',
                srcDark: 'img/logo-white.svg',
            },
            items: [
                {
                    href: "https://github.com/invergent-ai/surogate",
                    position: "right",
                    label: "GitHub",
                },
                {
                    href: "https://x.com/surogate_ai",
                    position: "right",
                    label: "Follow us on X",
                },
                {
                    type: 'search',
                    position: 'right',
                }
            ],
        },
        algolia: {
          appId: 'K46M9C3QXR',
          apiKey: '8697cfccdeb5371226efca0dec10e96f',
          indexName: 'docs_surogate_ai_k46m9c3qxr_pages',
        },
        footer: {
            style: 'dark',
            links: [],
            copyright: `Copyright © ${new Date().getFullYear()} Invergent. All rights reserved.`,
        }
    },
    headTags: [
       {
            tagName: 'meta',
            attributes: { name: 'keywords', content: 'LLM, training, fine-tuning, NVIDIA, Blackwell, B200, FP8, NVFP4, QLoRA, CUDA, high-performance, AI, SM100, SM120, RTX 5090, pre-training, Surogate' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'description', content: 'Browse tutorials and guides, learn how Surogate works and get started quickly with your first model.' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:type', content: 'website' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:url', content: 'https://docs.surogate.ai' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:title', content: 'Surogate Documentation' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:description', content: 'Insanely fast LLM pre-training and fine-tuning for modern NVIDIA GPUs' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:image', content: 'https://docs.surogate.ai/img/social-card.jpg' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:image:secure_url', content: 'https://docs.surogate.ai/img/social-card.jpg' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:image:alt', content: 'Surogate' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:locale', content: 'en_US' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'og:image:type', content: 'image/jpeg' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:site', content: 'surogate_ai' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:creator', content: 'surogate_ai' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:card', content: 'summary_large_image' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:url', content: 'https://docs.surogate.ai' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:title', content: 'Surogate Documentation' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:description', content: 'Insanely fast LLM pre-training and fine-tuning for modern NVIDIA GPUs' }
        },
        {
            tagName: 'meta',
            attributes: { name: 'twitter:image', content: 'https://docs.surogate.ai/img/social-card.jpg' }
        }
    ],
    presets: [
        [
            '@docusaurus/preset-classic',
            {
                docs: {
                    /* other docs plugin options */
                    routeBasePath: '/', // Serve the docs at the site's root
                    sidebarPath: require.resolve('./sidebars.js'),
                    remarkPlugins: [remarkMath],
                    rehypePlugins: [rehypeKatex],
                },
                blog: false, // Optional: disable the blog plugin
                theme: {
                    customCss: [require.resolve("./src/css/custom.css")],
                },
                sitemap: {
                    lastmod: 'date',
                    changefreq: 'weekly',
                },
                gtag: {
                    trackingID: 'G-7YRDGYZN0R',
                }
            },
        ],
    ],
    stylesheets: [
        {
          href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
          type: 'text/css',
          integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
          crossorigin: 'anonymous',
        },
    ],
    plugins: [
    ],
    markdown: {
        format: 'detect',
    }
};
