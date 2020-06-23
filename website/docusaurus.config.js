/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.
const baseUrl = "/";

// List of projects/orgs using your project for the users page.
const users = [];

module.exports = {
  // ...
  title: 'Classy Vision',
  tagline: 'An end-to-end framework for image and video classification',
  url: 'https://classyvision.ai', // Your website URL
  baseUrl: '/',
  organizationName: 'facebookresearch',
  projectName: 'ClassyVision',
  favicon: 'img/favicon.png',
  scripts: [
      'https://buttons.github.io/buttons.js',
      // Copy-to-clipboard button for code blocks
      `${baseUrl}js/code_block_buttons.js`,
  ],

  presets: [
    [
      '@docusaurus/preset-classic',
      {
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        // ...
      },
    ],
  ],

  themeConfig: {
    disableDarkMode: true,
    navbar: {
      textColor: '#091E42',
      logo: {
        alt: 'Classy Vision',
        src: 'img/cv-logo.png',
      },
      links: [
        { href: `${baseUrl}tutorials/`, label: 'Tutorials', position: 'left'},
        { href: `${baseUrl}api/`, label: 'API Reference' , position: 'left'},
        {
          href: 'https://github.com/facebookresearch/ClassyVision',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
      footer: {
      logo: {
        alt: 'Facebook Open Source Logo',
        src: 'https://docusaurus.io/img/oss_logo.png',
        href: 'https://opensource.facebook.com/',
      },
      copyright: `Copyright © ${new Date().getFullYear()} Facebook, Inc.`,
    },
    image: 'img/docusaurus.png',
    // Equivalent to `docsSideNavCollapsible`.
    sidebarCollapsible: false,
  },

  // CSS sources to load
  stylesheets: [`${baseUrl}css/code_block_buttons.css`],
};
