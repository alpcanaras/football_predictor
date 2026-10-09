// The one working screen: fixtures on the left, the coupon on the right.
import { h } from './ui.js';
import { FixturesView } from './fixtures.js';
import { SlipView } from './slip.js';

export class BuildPage {
  constructor(root, coupon) {
    const fx = h('section', { class: 'fixtures' });
    const slip = h('aside', { class: 'slip', 'aria-label': 'Your coupon' });
    root.replaceChildren(h('div', { class: 'build' }, fx, slip));
    this.fx = new FixturesView(fx, coupon);
    this.slip = new SlipView(slip, coupon);
  }
  destroy() { this.fx.destroy(); this.slip.destroy(); }
}
