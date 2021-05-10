# GARIGO
#### make face of un-related people mosaic in video
#### cooperate with apeltop, lim8540

## what
* when capturing or broadcasting outside, it can be possible that people(or their face) show in the program but they are not related in this and it woulb be issue between common-person and creator
* so, this project wants to show related actor only in program or video

## static, not dynamic(real-time)
* this project can be applied to static video or image
* due to speed and performance, this is not appropriate for realtime video

## the way to make better performance
* compare with frames whose similarity is high(in this code, similarity over 0.9(or 0.8) is regard as similar)
* up-scaling
* argument of detection
  * model=cnn (<> hog)
  * num of upscaling = 2 (default 1)
* argument of recognition
  * num_jitters=5 (default 1, when number up, times up)
  * model="large" (<> small)

## performance
![슬라이드53](https://user-images.githubusercontent.com/45033215/117620636-a030aa80-b1ab-11eb-960a-e6d4b08b60ba.PNG)

## face detection and recognition
https://pypi.org/project/face-recognition/

https://github.com/ageitgey/face_recognition

## project presentation
![슬라이드1](https://user-images.githubusercontent.com/45033215/117620374-695a9480-b1ab-11eb-8538-d286221eaa19.PNG)
![슬라이드2](https://user-images.githubusercontent.com/45033215/117620377-6a8bc180-b1ab-11eb-9890-e4b71c8861b6.PNG)
![슬라이드3](https://user-images.githubusercontent.com/45033215/117620379-6a8bc180-b1ab-11eb-952d-9f85c895e368.PNG)
![슬라이드4](https://user-images.githubusercontent.com/45033215/117620381-6b245800-b1ab-11eb-9585-5507d3aae2ed.PNG)
![슬라이드5](https://user-images.githubusercontent.com/45033215/117620384-6bbcee80-b1ab-11eb-943b-d625253559d7.PNG)
![슬라이드6](https://user-images.githubusercontent.com/45033215/117620387-6bbcee80-b1ab-11eb-8e27-439c844c5202.PNG)
![슬라이드7](https://user-images.githubusercontent.com/45033215/117620388-6c558500-b1ab-11eb-8a6d-b32c91500dd9.PNG)
![슬라이드8](https://user-images.githubusercontent.com/45033215/117620389-6c558500-b1ab-11eb-8c63-c23ee2ba067d.PNG)
![슬라이드9](https://user-images.githubusercontent.com/45033215/117620392-6cee1b80-b1ab-11eb-8df9-90ae5225aca4.PNG)
![슬라이드10](https://user-images.githubusercontent.com/45033215/117620395-6d86b200-b1ab-11eb-9e28-3cf5f56bcea4.PNG)
![슬라이드11](https://user-images.githubusercontent.com/45033215/117620397-6d86b200-b1ab-11eb-8937-c8015ccae4a0.PNG)
![슬라이드12](https://user-images.githubusercontent.com/45033215/117620399-6e1f4880-b1ab-11eb-9730-a8f1fc9e7c7f.PNG)
![슬라이드13](https://user-images.githubusercontent.com/45033215/117620400-6eb7df00-b1ab-11eb-858d-e276dbd6bc10.PNG)
![슬라이드14](https://user-images.githubusercontent.com/45033215/117620402-6eb7df00-b1ab-11eb-88e7-e753660a532a.PNG)
![슬라이드15](https://user-images.githubusercontent.com/45033215/117620404-6f507580-b1ab-11eb-91d1-0efe47f1fb4c.PNG)
![슬라이드16](https://user-images.githubusercontent.com/45033215/117620407-6f507580-b1ab-11eb-8783-68007f9764b1.PNG)
![슬라이드17](https://user-images.githubusercontent.com/45033215/117620408-6fe90c00-b1ab-11eb-9af7-b5b92aed3ace.PNG)
![슬라이드18](https://user-images.githubusercontent.com/45033215/117620410-6fe90c00-b1ab-11eb-85b7-a087c9043417.PNG)
![슬라이드19](https://user-images.githubusercontent.com/45033215/117620413-7081a280-b1ab-11eb-860d-3bdc5d4d936b.PNG)
![슬라이드20](https://user-images.githubusercontent.com/45033215/117620414-7081a280-b1ab-11eb-9bd7-089f1f814438.PNG)
![슬라이드21](https://user-images.githubusercontent.com/45033215/117620417-711a3900-b1ab-11eb-8226-52e87981ebf7.PNG)
![슬라이드22](https://user-images.githubusercontent.com/45033215/117620420-711a3900-b1ab-11eb-98b3-2b11f6cf7315.PNG)
![슬라이드23](https://user-images.githubusercontent.com/45033215/117620422-71b2cf80-b1ab-11eb-9165-599b71328e4f.PNG)
![슬라이드24](https://user-images.githubusercontent.com/45033215/117620424-724b6600-b1ab-11eb-92b7-63370ee12c6b.PNG)
![슬라이드25](https://user-images.githubusercontent.com/45033215/117620426-724b6600-b1ab-11eb-8f21-8a682c9a9a3f.PNG)
![슬라이드26](https://user-images.githubusercontent.com/45033215/117620427-72e3fc80-b1ab-11eb-92c7-6d1ae2ceaee5.PNG)
![슬라이드27](https://user-images.githubusercontent.com/45033215/117620429-72e3fc80-b1ab-11eb-92ba-903077f63859.PNG)
![슬라이드28](https://user-images.githubusercontent.com/45033215/117620431-737c9300-b1ab-11eb-8ea3-1dcf24ebe615.PNG)
![슬라이드29](https://user-images.githubusercontent.com/45033215/117620434-737c9300-b1ab-11eb-8c85-7b15c8133cad.PNG)
![슬라이드30](https://user-images.githubusercontent.com/45033215/117620435-74152980-b1ab-11eb-8bd6-2f60df429d7a.PNG)
![슬라이드31](https://user-images.githubusercontent.com/45033215/117620436-74adc000-b1ab-11eb-865b-61e819a4cff3.PNG)
![슬라이드32](https://user-images.githubusercontent.com/45033215/117620439-74adc000-b1ab-11eb-8a50-afe8f113ab96.PNG)
![슬라이드33](https://user-images.githubusercontent.com/45033215/117620442-75465680-b1ab-11eb-847e-635a2883c4a8.PNG)
![슬라이드34](https://user-images.githubusercontent.com/45033215/117620444-75465680-b1ab-11eb-9673-a03e7eae66bc.PNG)
![슬라이드35](https://user-images.githubusercontent.com/45033215/117620448-75deed00-b1ab-11eb-83dd-cee7b7f5253b.PNG)
![슬라이드36](https://user-images.githubusercontent.com/45033215/117620451-76778380-b1ab-11eb-954e-3378b316101a.PNG)
![슬라이드37](https://user-images.githubusercontent.com/45033215/117620453-76778380-b1ab-11eb-8978-109af253872e.PNG)
![슬라이드38](https://user-images.githubusercontent.com/45033215/117620459-77101a00-b1ab-11eb-9588-eb48d567b95b.PNG)
![슬라이드39](https://user-images.githubusercontent.com/45033215/117620460-77101a00-b1ab-11eb-8d22-b1fd13ba2a7d.PNG)
![슬라이드40](https://user-images.githubusercontent.com/45033215/117620465-77a8b080-b1ab-11eb-8670-c2bd45f8f139.PNG)
![슬라이드41](https://user-images.githubusercontent.com/45033215/117620467-78414700-b1ab-11eb-86f9-e87e93c9f58b.PNG)
![슬라이드42](https://user-images.githubusercontent.com/45033215/117620468-78414700-b1ab-11eb-92df-57a2275d2175.PNG)
![슬라이드43](https://user-images.githubusercontent.com/45033215/117620470-78d9dd80-b1ab-11eb-876c-c6206befe0a2.PNG)
![슬라이드44](https://user-images.githubusercontent.com/45033215/117620472-78d9dd80-b1ab-11eb-9cb4-f9e1c73fd2cb.PNG)
![슬라이드45](https://user-images.githubusercontent.com/45033215/117620475-79727400-b1ab-11eb-8b6c-916def7a9929.PNG)
![슬라이드46](https://user-images.githubusercontent.com/45033215/117620477-79727400-b1ab-11eb-8122-a873119e1be6.PNG)
![슬라이드47](https://user-images.githubusercontent.com/45033215/117620478-7a0b0a80-b1ab-11eb-8566-76f0a46f6ed9.PNG)
![슬라이드48](https://user-images.githubusercontent.com/45033215/117620480-7a0b0a80-b1ab-11eb-9656-15d0d5c6e4e3.PNG)
![슬라이드49](https://user-images.githubusercontent.com/45033215/117620482-7aa3a100-b1ab-11eb-824f-25ebf028995f.PNG)
![슬라이드50](https://user-images.githubusercontent.com/45033215/117620486-7aa3a100-b1ab-11eb-8bd4-1e2e50610c59.PNG)
![슬라이드51](https://user-images.githubusercontent.com/45033215/117620489-7b3c3780-b1ab-11eb-954d-7407361d1ca9.PNG)
![슬라이드52](https://user-images.githubusercontent.com/45033215/117620490-7bd4ce00-b1ab-11eb-9423-2c1010841f8d.PNG)
![슬라이드53](https://user-images.githubusercontent.com/45033215/117620492-7bd4ce00-b1ab-11eb-82b4-8108441b4e4f.PNG)
![슬라이드54](https://user-images.githubusercontent.com/45033215/117620495-7c6d6480-b1ab-11eb-9a1e-3a6b8e82ef63.PNG)
![슬라이드55](https://user-images.githubusercontent.com/45033215/117620496-7c6d6480-b1ab-11eb-893a-91978abebed5.PNG)
![슬라이드56](https://user-images.githubusercontent.com/45033215/117620499-7d05fb00-b1ab-11eb-9d3a-ab08855fb255.PNG)
![슬라이드57](https://user-images.githubusercontent.com/45033215/117620502-7d05fb00-b1ab-11eb-871e-7dd08f7b54ae.PNG)
![슬라이드58](https://user-images.githubusercontent.com/45033215/117620503-7d9e9180-b1ab-11eb-8ead-d219d6ca9bb1.PNG)
![슬라이드59](https://user-images.githubusercontent.com/45033215/117620506-7e372800-b1ab-11eb-8daf-8ebb0d7181db.PNG)
![슬라이드60](https://user-images.githubusercontent.com/45033215/117620508-7e372800-b1ab-11eb-8434-94134c20c24d.PNG)
![슬라이드61](https://user-images.githubusercontent.com/45033215/117620510-7ecfbe80-b1ab-11eb-9952-1ed7d198fcf9.PNG)


