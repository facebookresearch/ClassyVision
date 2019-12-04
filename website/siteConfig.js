/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// See https://docusaurus.io/docs/site-config for all the possible
// site configuration options.

// Define this so it can be easily modified in scripts (to host elsewhere)
const baseUrl = "/";

// List of projects/orgs using your project for the users page.
const users = [];

const siteConfig = {
  title: 'Classy Vision', // Title for your website.
  tagline: 'An end-to-end framework for image and video classification',
  url: 'https://classyvision.ai', // Your website URL
  baseUrl: baseUrl, // Base URL for your project */
  // For github.io type URLs, you would set the url and baseUrl like:
  //   url: 'https://facebook.github.io',
  //   baseUrl: '/test-site/',

  // Used for publishing and more
  projectName: 'ClassyVision',
  organizationName: 'facebookresearch',
  // For top-level user or org sites, the organization is still the same.
  // e.g., for the https://JoelMarcey.github.io site, it would be set like...
  //   organizationName: 'JoelMarcey'

  // For no header links in the top nav bar -> headerLinks: [],
  headerLinks: [
    { href: `${baseUrl}tutorials/`, label: "Tutorials" },
    { href: `${baseUrl}api/`, label: "API Reference" },
    { href: "https://github.com/facebookresearch/ClassyVision", label: "GitHub" },
    { search: true } // position search box to the very right
  ],

  // If you have users set above, you add it here:
  users,

  /* path to images for header/footer */
  headerIcon: 'img/favicon.png',
  //footerIcon: 'img/favicon.ico',
  favicon: 'img/favicon.png',
  logo: 'img/cv-logo.png',

  // colors for website
  colors: {
    primaryColor: "#f29837", // orange
    secondaryColor: "#f0bc40" // yellow
  },

  /* Custom fonts for website */
  /*
  fonts: {
    myFont: [
      "Times New Roman",
      "Serif"
    ],
    myOtherFont: [
      "-apple-system",
      "system-ui"
    ]
  },
  */

  // This copyright info is used in /core/Footer.js and blog RSS/Atom feeds.
  copyright: `Copyright \u{00A9} ${new Date().getFullYear()} Facebook Inc.`,

  highlight: {
    // Highlight.js theme to use for syntax highlighting in code blocks.
    theme: 'default',
  },

  // Add custom scripts here that would be placed in <script> tags.
  scripts: [
      'https://buttons.github.io/buttons.js',
      // Copy-to-clipboard button for code blocks
      `${baseUrl}js/code_block_buttons.js`,
  ],

  // CSS sources to load
  stylesheets: [`${baseUrl}css/code_block_buttons.css`],

  // On page navigation for the current documentation page.
  onPageNav: 'separate',
  // No .html extensions for paths.
  cleanUrl: true,

  // Open Graph and Twitter card images.
  ogImage: 'img/undraw_online.svg',
  twitterImage: 'img/undraw_tweetstorm.svg',

  wrapPagesHTML: true,

  // For sites with a sizable amount of content, set collapsible to true.
  // Expand/collapse the links and subcategories under categories.
  // docsSideNavCollapsible: true,

  // Show documentation's last contributor's name.
  // enableUpdateBy: true,

  // Show documentation's last update time.
  // enableUpdateTime: true,

  // You may provide arbitrary config keys to be used as needed by your
  // template. For example, if you need your repo's URL...
  //   repoUrl: 'https://github.com/facebook/test-site',
};

module.exports = siteConfig;
