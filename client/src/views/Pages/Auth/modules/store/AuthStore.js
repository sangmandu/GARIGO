import {action, computed, flow, makeAutoObservable, observable} from 'mobx';
import AuthRepository from '../repository/AuthRepository';

let authRepository = new AuthRepository();

class AuthStore {
	@observable jwtToken = '1';

	@computed get isLogin() {
		return !!this.jwtToken;
	}

	constructor() {
		makeAutoObservable(this);
	}

	signIn = flow(function* (id, pw) {
		console.log('sign in AuthStore');
		const {data, status} = yield authRepository.signIn(id, pw);

		if (status !== 200) {
			return;
		}

		this.jwtToken = data;
	});
}

const store = new AuthStore();
export default store;
