import axios from 'axios';

class AuthRepository {
	HOST = 'http://127.0.0.1:8000';
	URL = this.HOST + '/auth';

	async signIn(id, pw) {
		try {
			console.log(this.URL + `?id=${id}&pw=${pw}`);
			return axios.get(`${this.URL}?id=${id}&pw=${pw}`);
		} catch (e) {
			console.log(e);
		}
	}

}

export default AuthRepository;
