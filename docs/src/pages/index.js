import React, { useEffect } from 'react';
import { useHistory } from '@docusaurus/router';

const HomePageRedirect = () => {
  const history = useHistory();

  useEffect(() => {
    history.push('/2025-1B-T12-EC06-G03/introducao');
  }, [history]);

  return null;
};

export default HomePageRedirect;
