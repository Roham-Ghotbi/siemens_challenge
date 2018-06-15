/*
 * Requires:
 *     psiturk.js
 *     utils.js
 */

// Initalize psiturk object
var psiTurk = new PsiTurk(uniqueId, adServerLoc, mode);
coach_on = true

var mycondition = condition;  // these two variables are passed by the psiturk server process
var mycounterbalance = counterbalance;  // they tell you which condition you have been assigned to
// they are not used in the stroop code but may be useful to you

// All pages to be loaded
var pages = [
	"video_stream.html"
];

psiTurk.preloadPages(pages);


var instructionPages = [ // add as a list as many pages as you like
	// "instructions/instruct-1.html",
	// "instructions/instruct-2.html",
	// "izzy_sim.html",
	// "instructions/instruct-3.html",
	// "instructions/instruct-4.html",
	"video_stream.html"
	
];



/********************
* HTML manipulation
*
* All HTML files in the templates directory are requested 
* from the server when the PsiTurk object is created above. We
* need code to get those pages from the PsiTurk object and 
* insert them into the document.
*
********************/


questions = false
var CarExperiment = function() {

	questions = true
	
};

wkrID =  psiTurk.taskdata.get('workerId')

/****************
* Questionnaire *
****************/

var Questionnaire = function() {

	var error_message = "<h1>Oops!</h1><p>Something went wrong submitting your HIT. This might happen if you lose your internet connection. Press the button to resubmit.</p><button id='resubmit'>Resubmit</button>";
	var responses = []; // create an empty array
	responses.push({
		key: "ID",
		value: wkrID
	})
	
	record_responses = function() {

		psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'submit'});

		$('textarea').each( function(i, val) {
			psiTurk.recordUnstructuredData(this.id, this.value);
		});
		$('select').each( function(i, val) {
			responses.push({
				key:   this.id,
				value: this.value
			});

			psiTurk.recordUnstructuredData(this.name, this.value);		
		});

	};

	prompt_resubmit = function() {
		replaceBody(error_message);
		$("#resubmit").click(resubmit);
	};

	resubmit = function() {
		replaceBody("<h1>Trying to resubmit...</h1>");
		reprompt = setTimeout(prompt_resubmit, 10000);
		
		psiTurk.saveData({
			success: function() {
			    clearInterval(reprompt); 
                psiTurk.computeBonus('compute_bonus', function(){finish()}); 
			}, 
			error: prompt_resubmit
		});
	};

	// Load the questionnaire snippet 
	
	psiTurk.showPage('postquestionnaire.html');

	psiTurk.recordTrialData({'phase':'postquestionnaire', 'status':'begin'});
	
	$("#next").click(function () {
	    record_responses();
	    $.ajax('http://0.0.0.0:5000/save_data', {
                type: "GET",
                data: responses
                });



	    psiTurk.saveData({
            success: function(){
                psiTurk.computeBonus('compute_bonus', function() { 
                	psiTurk.completeHIT(); // when finished saving compute bonus, the quit
                }); 
            }, 
            error: prompt_resubmit});
	});
    
	
};

// Task object to keep track of the current phase
var currentview;

/*******************
 * Run Task
 ******************/
$(window).load( function(){

	psiTurk.doInstructions(
		instructionPages, // a list of pages you want to display in sequence
		function() { currentview = new Questionnaire(); } 
	);	
    
    //psiTurk.doInstructions(
    //	gamePage
    //);
    //psiTurk.showPage('winter_game.html');

});
