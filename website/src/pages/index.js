/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

import React from 'react';
import classnames from 'classnames';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';

const features = [
  {
    title: <>Scalable</>,
    // imageUrl: 'img/expanding_arrows.svg',
    description: (
      <>
        Launch large distributed training jobs with minimal effort.
        No need for proprietary infrastructure.
      </>
    ),
  },
  {
    title: <>Built on PyTorch</>,
    // imageUrl: 'img/pytorch_logo.svg',
    description: (
      <>
        Supports most types of PyTorch models and can be used with
        minimal modification to the original neural network.
      </>
    ),
  },
  {
    title: <>Plug and play</>,
    // imageUrl: 'img/modular.svg',
    description: (
      <>
        Open source, modular API for computer vision research.
        Everyone is welcome to contribute.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={classnames('col col--3', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="An end-to-end framework for image and video classification">
      <header className={classnames('hero hero--primary', styles.heroBanner)}>
        <div className="container">
          <img src='img/cv-logo.png' alt="ClassyVision logo." width="100"/>
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <Link
              className={classnames(
                'button button--outline button--secondary button--lg',
                styles.getStarted,
              )}
              to={'#quickstart'}>
              Get Started
            </Link>


          </div>
        </div>
      </header>
      <main>
        {features && features.length && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map(({title, imageUrl, description}) => (
                  <Feature
                    key={title.id}
                    title={title}
                    imageUrl={imageUrl}
                    description={description}
                  />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
