/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock;
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="splashLogo">
        <img src={props.img_src} alt="Project Logo" className="primaryLogoImage"/>
      </div>
    );

    const ProjectTitle = () => (
      <h2 className="projectTitle">
        <small>{siteConfig.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={siteConfig.logo} />
        <div className="inner">
          <ProjectTitle siteConfig={siteConfig} />
          <PromoSection>
            <Button href={'#quickstart'}>Get Started</Button>
            <Button href={`${baseUrl}tutorials/`}>Tutorials</Button>
            <Button href={"https://github.com/facebookresearch/ClassyVision"}>GitHub</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    const Description = () => (
      <Block background="light">
        {[
          {
            content:
              'This is another description of how this project is useful',
            image: `${baseUrl}img/cv-icon.png`,
            imageAlign: 'right',
            title: 'Description',
          },
        ]}
      </Block>
    );
    // getStartedSection
    const pre = '```';
    // Example for model fitting
    const createProject = `${pre}bash
classy-project my-project
cd my-project
    `;
    //
    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <h4>Install Classy Vision:</h4>
              <a>via pip:</a>
              <MarkdownBlock>{bash`pip install classy_vision`}</MarkdownBlock>
            </li>
            <li>
              <h4>Create your first project:</h4>
              <MarkdownBlock>{createProject}</MarkdownBlock>
            </li>
            <li>
              <h4>Start training!</h4>
              <MarkdownBlock>{bash`./classy_train.py --config configs/template_config.json`}</MarkdownBlock>
            </li>
          </ol>
        </Container>
      </div>
    );

    const Features = () => (
    <div className="productShowcaseSection" style={{textAlign: 'center'}}>
      <Block layout="threeColumn">
        {[
          {
            content: 'Launch large distributed training jobs with minimal effort. No need for proprietary infrastructure.',
            image: `${baseUrl}img/expanding_arrows.svg`,
            imageAlign: 'top',
            title: 'Scalable',
          },
          {
            content:
              'Supports most types of PyTorch models and can be used with minimal modification to the original neural network.',
            image: `${baseUrl}img/pytorch_logo.svg`,
            imageAlign: 'top',
            title: 'Built on PyTorch',
          },
          {
            content:
              'Open source, modular API for computer vision research. Everyone is welcome to contribute.',
            image: `${baseUrl}img/modular.svg`,
            imageAlign: 'top',
            title: 'Plug and play',
          },
        ]}
      </Block>
    </div>
  );

    const Showcase = () => {
      if ((siteConfig.users || []).length === 0) {
        return null;
      }

      const showcase = siteConfig.users
        .filter(user => user.pinned)
        .map(user => (
          <a href={user.infoLink} key={user.infoLink}>
            <img src={user.image} alt={user.caption} title={user.caption} />
          </a>
        ));

      const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;

      return (
        <div className="productShowcaseSection paddingBottom">
          <h2>Who is Using This?</h2>
          <p>This project is used by all these people</p>
          <div className="logos">{showcase}</div>
          <div className="more-users">
            <a className="button" href={pageUrl('users.html')}>
              More {siteConfig.title} Users
            </a>
          </div>
        </div>
      );
    };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="landingPage mainContainer">
          <Features />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
