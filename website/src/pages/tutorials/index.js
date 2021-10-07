/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">Classy Tutorials</h1>
            </header>
            <body>
              <p>
                The tutorials here will help you understand and use Classy Vision. They assume that you are familiar with PyTorch and its basic features.
              </p>
              <p>
                If you are new to Classy Vision, the easiest way to get started is
                with the{' '}
                <a href="getting_started">
                  Getting started with Classy Vision
                </a>{' '}
                tutorial.
              </p>
              <p>
                If you are new to PyTorch, the easiest way to get started is
                with the{' '}
                <a href="https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py">
                  What is PyTorch?
                </a>{' '}
                tutorial.
              </p>
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
