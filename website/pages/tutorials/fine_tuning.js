const CWD = process.cwd();

const React = require('react');
const Tutorial = require(`${CWD}/core/Tutorial.js`);

class TutorialPage extends React.Component {
  render() {
      const {config: siteConfig} = this.props;
      const {baseUrl} = siteConfig;
      return <Tutorial baseUrl={baseUrl} tutorialID="fine_tuning"/>;
  }
}

module.exports = TutorialPage;

